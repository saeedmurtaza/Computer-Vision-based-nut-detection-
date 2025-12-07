# app/camera/camera_worker.py
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Full, Empty
from threading import Thread, Event, Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pypylon import genicam

from ..sensor.serial_sensor import sensor_init, read_proximity_json
from ..core.config import (
    PLC_IP,
    PLC_PORT,
    PLC_BIT,
    GAMMA_DEFAULT,
    CLAHE_CLIP_DEFAULT,
    CLAHE_TILE_DEFAULT,
    DO_NORM_DEFAULT,
    DO_RL_DEFAULT,
    RL_ITER_DEFAULT,
    RL_PSF_SIZE_DEFAULT,
    SPEC_SUPPRESS_DEFAULT,
    SPEC_THR_DEFAULT,
    SPEC_MAXAREA_DEFAULT,
    PREVIEW_DEB_DEFAULT,
)
from ..core.utils import slugify_model_name
from ..detection.classify import detect_classes_from_image
from ..detection.yolo_wrapper import RealYOLO
from .roi import crop_bottom_half
from .enhancer import beautify_preview, load_sr_model
from .basler_driver import BaslerCamera
from ..plc.plc_client import PLCClient


class ProximityCameraTrigger:
    """
    Continuous preview stream, YOLO runs ONCE per proximity rising edge.
    Saves annotated frame for every trigger (pass/reject), focused by ROI if configured.
    """

    def __init__(
        self,
        save_dir: str,
        device,
        device_name: str = "Basler",
        torch_device: str = "cpu",
        model_path: Optional[str] = None,
        plc_ip: str = PLC_IP,
        plc_port: int = PLC_PORT,
        trigger_bit: str = PLC_BIT,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        camera_params: Optional[Dict[str, Any]] = None,
        gamma: float = GAMMA_DEFAULT,
        clahe_clip: float = CLAHE_CLIP_DEFAULT,
        clahe_tile: int = CLAHE_TILE_DEFAULT,
        do_norm: bool = DO_NORM_DEFAULT,
        do_rl: bool = DO_RL_DEFAULT,
        rl_iter: int = RL_ITER_DEFAULT,
        rl_psf_size: int = RL_PSF_SIZE_DEFAULT,
        spec_suppress: bool = SPEC_SUPPRESS_DEFAULT,
        spec_thr: int = SPEC_THR_DEFAULT,
        spec_maxarea: int = SPEC_MAXAREA_DEFAULT,
        preview_deblur: bool = PREVIEW_DEB_DEFAULT,
        conf: float = 0.95,
        iou: float = 0.30,
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "passes").mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "rejects").mkdir(parents=True, exist_ok=True)

        self.save_dir = save_dir
        self.device = device            # pypylon device info
        self.device_name = device_name
        self.torch_device = torch_device
        self.model_path = model_path

        # Basler wrapper
        self.basler: Optional[BaslerCamera] = None
        self.camera = None
        self.converter = None
        self._camera_lock = Lock()

        self.stop_event = Event()
        self.frame_queue: "Queue[np.ndarray]" = Queue(maxsize=10)
        self.frame_lock = Lock()
        self.latest_raw_image = None
        self.latest_det_input = None
        self.latest_annotated_image = None
        self.capture_on = False

        # Detection
        self.detector: Optional[RealYOLO] = None
        self.detect_enabled = True
        self.conf = conf
        self.iou = iou
        self.imgsz = 640
        # self.allowed_labels = ["Residue"]
        # self.neg_label = "Residue"     # dynamic label per camera for UI/logs
        # self.allowed_labels = ["NoThreads", "no_threads", "NoThread", "MissingThread", "ThreadMissing"]
        # self.neg_label = "NoThreads"
        self.fail_labels = ["Fail"]      # defect class(es) based on new non tab trained model
        self.pass_label = "Pass"         # good class
        self.neg_label = "Fail"          # for logging/UI “defect label”
        self.last_defects = 0

        # Enhancement
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.do_norm = do_norm
        self.do_rl = do_rl
        self.rl_iter = rl_iter
        self.rl_psf_size = rl_psf_size
        self.spec_suppress = spec_suppress
        self.spec_thr = spec_thr
        self.spec_maxarea = spec_maxarea
        self.preview_deblur = preview_deblur

        # Counters
        self.total_count = 0
        self.pass_count = 0
        self.reject_count = 0

        self.last_nuts = 0
        self.last_residue = 0
        self.last_state = "idle"
        self.last_trigger_ts = ""
        self.last_save_path = ""

        self.camera_params = camera_params or {
            "ExposureTime": 2000.0,
            "Gain": 6.0,
            "PixelFormat": "Mono8",
            "AcquisitionFrameRateEnable": True,
            "AcquisitionFrameRate": 20,
        }

        self.roi_norm: Optional[Tuple[float, float, float, float]] = roi_norm

        # Threads
        self.grab_thread: Optional[Thread] = None
        self.trigger_thread: Optional[Thread] = None

        self.pending_detection = Event()
        self._last_fire_ts = 0.0
        self._trigger_fire_time: float = 0.0
        self.trigger_count = 0
        self.cam_idx: int = 0  # set in start()

        # PLC (using PLCClient)
        self.plc_ip = plc_ip
        self.plc_port = plc_port
        self.trigger_bit = trigger_bit
        self.plc: Optional[PLCClient] = None

    # Problem type inference (Residue / Threads)
    # def set_problem_type_from_model(self, model_path: Optional[str]):
    #     name = (model_path or "").lower()
    #     if any(k in name for k in ("residue", "residues", "residue_defect")):
    #         self.allowed_labels = ["Residue", "residue"]
    #         self.neg_label = "Residue"
    #     elif any(k in name for k in ("no_threads", "nothreads", "no-threads", "missing_thread", "thread_missing")):
    #         self.allowed_labels = ["NoThreads", "no_threads", "NoThread", "MissingThread", "ThreadMissing"]
    #         self.neg_label = "NoThreads"

    # PLC
    def _connect_plc(self) -> bool:
        try:
            self.plc = PLCClient(self.plc_ip, self.plc_port)
            print(f"✅ PLC connected ({self.plc_ip}:{self.plc_port}), trigger bit {self.trigger_bit}")
            return True
        except Exception as e:
            print(f"⚠️ PLC connection failed: {e}")
            self.plc = None
            return False

    # Camera setup
    def setup_camera(self) -> bool:
        try:
            self.basler = BaslerCamera(self.device, name=self.device_name)
            if not self.basler.open():
                return False

            # keep a direct reference for param application
            self.camera = self.basler.camera
            self.converter = self.basler.converter

            if self.camera:
                # basic geometry & buffers
                self.camera.Width.Value = 2560
                self.camera.Height.Value = 2560
                self.camera.MaxNumBuffer = 10
                print(f"✅ Camera opened: {self.device_name} ({self.device.GetSerialNumber()})")
                print(f"TriggerMode: {self.camera.TriggerMode.Value}")
                print(f"AcquisitionMode: {self.camera.AcquisitionMode.Value}")

            self.apply_params()
            return True
        except genicam.GenericException as e:
            print(f"❌ Camera setup failed for {self.device_name}: {e}")
            self.basler = None
            self.camera = None
            self.converter = None
            return False

    def setup_yolo(self, model_path: str, conf: Optional[float] = None, iou: Optional[float] = None):
        self.set_problem_type_from_model(model_path)
        self.detector = RealYOLO(
            model_path,
            conf if conf is not None else self.conf,
            iou if iou is not None else self.iou,
            self.torch_device,
            allowed_labels=self.allowed_labels,
        )
        self.model_path = model_path
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        print(f"✅ YOLO loaded on {self.device_name} [{self.torch_device}] (decision on {self.neg_label}, conf={self.conf:.2f})")

    def set_camera_parameters(self, **params):
        for k, v in params.items():
            self.camera_params[k] = v
        self.apply_params()

    def apply_params(self):
        if not (self.camera and self.camera.IsOpen()):
            return
        try:
            nm = self.camera.GetNodeMap()
            for auto_node in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
                try:
                    node = nm.GetNode(auto_node)
                    if node and genicam.IsWritable(node):
                        node.FromString("Off")
                except Exception:
                    pass
            try:
                node_pf = nm.GetNode("PixelFormat")
                if node_pf and genicam.IsWritable(node_pf):
                    node_pf.FromString(str(self.camera_params.get("PixelFormat", "Mono8")))
            except Exception:
                pass
            for k, v in self.camera_params.items():
                try:
                    node = nm.GetNode(k)
                    if node and genicam.IsWritable(node):
                        try:
                            node.SetValue(float(v) if isinstance(v, (int, float)) else v)
                        except Exception:
                            node.FromString(str(v))
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Param apply failed: {e}")

    # Saving outcome
    def _save_outcome(self, cam_idx: int, img_bgr: np.ndarray, is_reject: bool, model_token: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        subdir = "rejects" if is_reject else "passes"
        name = ("reject" if is_reject else "pass") + f"_cam{cam_idx}_{model_token}_{stamp}.jpg"
        out_path = str(Path(self.save_dir) / subdir / name)
        try:
            cv2.imwrite(out_path, img_bgr)
            print(f"⚠️ --------------Save succeed-----------------: {out_path}")
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
        return out_path

    # One-shot detection (Residue logic preserved, now using PLCClient)
    def _run_one_shot_detection(self, det_input_bgr: np.ndarray, cam_idx: int):
        model_token = slugify_model_name(self.model_path)
        print("point inside 1-----------------------------------------------------")

        # Ensure uint8 + BGR
        det_input = det_input_bgr.astype(np.uint8)
        if len(det_input.shape) == 2:
            det_input = cv2.cvtColor(det_input, cv2.COLOR_GRAY2BGR)

        # Run YOLO detector → [(class_name, conf), ...]
        detected = detect_classes_from_image(det_input, conf=0.25, iou=0.7)
        print("------------------------------------what class--------------------------", detected)

        # Decide using label + confidence
        top_label = None
        top_conf = 0.0
        if detected:
            top_label, top_conf = detected[0]

        thr = 0.61
        is_reject = False
        is_pass   = False

        if top_label is None:
            # No detection → safest is REJECT
            is_reject = True
        else:
            if top_label == "Pass" and top_conf >= thr:
                is_pass = True
            elif top_label == "Fail" and top_conf >= thr:
                is_reject = True
            else:
                # Unknown label or low conf → treat as REJECT
                is_reject = True

        # Update counters / state
        self.total_count += 1
        self.last_defects = 1 if is_reject else 0
        self.last_nuts = 1 if is_pass else 0
        self.last_state = "pass" if is_pass else "reject"

        print("point inside 3-----------------------------------------------------")
        # is_reject controls passes/ rejects folders (name stays the same)
        self.last_save_path = self._save_outcome(cam_idx, det_input_bgr, is_reject, model_token)
        print("point inside 4-----------------------------------------------------")

        try:
            if is_pass:
                self.pass_count += 1

                # PLC PASS pulse (D7020)
                if self.plc:
                    self.plc.write_bit("D7020", "1")
                    time.sleep(0.5)
                    self.plc.write_bit("D7020", "0")

                print("---------------------pass_count---------", self.pass_count)
                print("----------------------total_count---------", self.trigger_count)
                print(
                    f"✅ PASS   @ {self.last_trigger_ts} :: "
                    f"trigger#{self.trigger_count}, Nuts=1, {self.neg_label}=0  "
                    f"saved→ {self.last_save_path}"
                )
            else:
                self.reject_count += 1
                print(
                    f"❌ REJECT @ {self.last_trigger_ts} :: "
                    f"trigger#{self.trigger_count}, {self.neg_label}={self.last_defects}  "
                    f"saved→ {self.last_save_path}"
                )
                print("----------------------reject_count---------", self.reject_count)
                print("----------------------total_count---------", self.trigger_count)

                # PLC NG pulse (D7010)
                if self.plc:
                    self.plc.write_bit("D7010", "1")
                    time.sleep(0.5)
                    self.plc.write_bit("D7010", "0")

        except Exception as e:
            # Any error in detection → treat as REJECT, pulse NG bit
            print("empty detection / error:", e)
            self.reject_count += 1
            print(
                f"❌ REJECT @ {self.last_trigger_ts} :: "
                f"trigger#{self.trigger_count}, {self.neg_label}={self.last_defects}  "
                f"saved→ {self.last_save_path}"
            )
            print("----------------------reject_count---------", self.reject_count)
            print("----------------------total_count---------", self.trigger_count)

            if self.plc:
                self.plc.write_bit("D7010", "1")
                time.sleep(0.5)
                self.plc.write_bit("D7010", "0")


    def take_image(self, cam_idx: int, sr):
        """
        Single-grab capture using BaslerCamera, then ROI crop + beautify + detection.
        """
        # default to self.cam_idx if caller passes something invalid
        try:
            cam_index = int(cam_idx)
        except Exception:
            cam_index = getattr(self, "cam_idx", 0)

        if not self.basler or not self.basler.is_open():
            print("⚠️ take_image() called but BaslerCamera not ready.")
            return

        try:
            print(datetime.now(), "initialized--------------------------------------------")

            with self._camera_lock:
                self.basler.start_single_grab()
                success, frame = self.basler.retrieve_one(timeout_ms=5000)

            if not success or frame is None:
                print("⚠️ Image capture failed / timeout")
                return

            print(datetime.now(), "frame retrieved--------------------------------------------")

            # frame is already BGR from BaslerCamera
            frame_resized = cv2.resize(frame, (self.imgsz, self.imgsz))
            print("---------------------this is reached------------------------")
            print(datetime.now(), "point 6------------------------------------------")
            crop_bottom = crop_bottom_half(frame_resized)
            print(datetime.now(), "point 7-------------------------------------------")
            preview_base = crop_bottom
            preview_img = beautify_preview(preview_base)
            det_input = preview_base

            with self.frame_lock:
                self.latest_raw_image = preview_img
                self.latest_det_input = det_input

            try:
                self.frame_queue.put_nowait(preview_img)
            except Full:
                pass

            print(datetime.now(), "point 12-------------------------------------------")
            self._run_one_shot_detection(det_input, cam_index)

        except Exception as e:
            print(f"⚠️ Image capture error: {e}")
        finally:
            try:
                if self.basler:
                    self.basler.stop_grab()
            except Exception:
                pass

    def _trigger_loop(self):
        """
        Poll Arduino proximity sensor via serial and run one-shot capture on rising edge.
        """
        ser = sensor_init()
        last_state = 1

        sr = load_sr_model("./ESPCN_x4.pb", scale=4)
        print("🔍 Listening proximity sensor (Arduino + serial)")
        if not self.setup_camera():
            ser.close()
            return

        self.capture_on = True
        while not self.stop_event.is_set():
            try:
                reading = read_proximity_json(ser)
                if reading is None:
                    time.sleep(0.0001)
                    continue

                state, count, analog = reading
                now = time.time()

                # Detect falling edge: from 1 -> 0 (following your original logic)
                camera_on = state == 0 and last_state == 1

                # For testing, you previously forced always True; keep behavior if needed:
                # camera_on = True

                if camera_on:
                    self.trigger_count += 1
                    self.last_trigger_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    print(f"🚨 Trigger #{self.trigger_count} ↑ → schedule one detection")
                    self._trigger_fire_time = now
                    self._last_fire_ts = now
                    self.take_image(self.cam_idx, sr)

                last_state = state
                time.sleep(0.0001)

            except Exception as e:
                print(f"⚠️ Trigger loop error: {e}")

        ser.close()

    # Lifecycle
    def start(self, cam_idx: int):
        self.cam_idx = cam_idx
        self._connect_plc()
        self.stop_event.clear()
        if not (self.trigger_thread and self.trigger_thread.is_alive()):
            self.trigger_thread = Thread(target=self._trigger_loop, daemon=True)
            self.trigger_thread.start()

    def stop(self):
        self.stop_event.set()
        self.capture_on = False
        if self.trigger_thread and self.trigger_thread.is_alive():
            self.trigger_thread.join(timeout=1.5)
        if self.grab_thread and self.grab_thread.is_alive():
            self.grab_thread.join(timeout=1.5)

        try:
            if self.basler:
                self.basler.close()
        except Exception:
            pass

        try:
            if self.plc and hasattr(self.plc, "close"):
                self.plc.close()
        except Exception:
            pass

    def capture_image(self):
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            with self.frame_lock:
                return self.latest_raw_image

#Test block

# if __name__ == "__main__":
#     """
#     Simple sanitytest for ProximityCameraTrigger.

#     - If a Basler camera is found:
#         * connects PLC (if available)
#         * sets up camera
#         * grabs one frame
#         * runs one-shot detection on that frame (errors are caught)
#         * saves an image into passes/ or rejects/

#     - If NO camera is found:
#         * creates a dummy gray image
#         * runs beautify + bottom-crop
#         * saves a dummy pass image (tests _save_outcome)
#     """

#     from pypylon import pylon
#     from app.core.config import SAVE_DIR, TORCH_DEVICE
#     from app.camera.enhancer import beautify_preview
#     from app.camera.roi import crop_bottom_half

#     print("\n=== ProximityCameraTrigger sanitytest ===")

#     tl_factory = pylon.TlFactory.GetInstance()
#     devices = tl_factory.EnumerateDevices()

#     # -------------------------
#     # No camera → dummy test
#     # -------------------------
#     if not devices:
#         print("⚠️ No Basler cameras detected → running dummy image test only.")

#         # 1) Create dummy image
#         dummy = np.full((640, 640, 3), 128, dtype=np.uint8)
#         cropped = crop_bottom_half(dummy)
#         preview = beautify_preview(cropped)

#         # 2) Instantiate trigger with dummy device
#         trig = ProximityCameraTrigger(
#             save_dir=SAVE_DIR,
#             device=None,
#             device_name="Dummy",
#             torch_device=TORCH_DEVICE,
#             model_path=None,
#         )

#         # 3) Test saving outcome (also tests folder creation)
#         out_path = trig._save_outcome(cam_idx=0, img_bgr=preview,
#                                       is_reject=False, model_token="dummy")
#         print(f"Dummy save OK → {out_path}")
#         print("=== sanitytest finished (no camera) ===\n")
#     else:
#         # -------------------------
#         # Real camera test
#         # -------------------------
#         dev = devices[0]
#         print(f"✅ Found {len(devices)} camera(s). Using index 0: {dev.GetFriendlyName()}")

#         trig = ProximityCameraTrigger(
#             save_dir=SAVE_DIR,
#             device=dev,
#             device_name=dev.GetFriendlyName(),
#             torch_device=TORCH_DEVICE,
#             model_path=None,  # you can point this later to your best.pt
#         )

#         # 1) PLC connection (will just print a warning if driver not available)
#         print("\n[1] Testing PLC connection …")
#         trig._connect_plc()

#         # 2) Camera setup + parameter application
#         print("[2] Testing camera setup …")
#         if not trig.setup_camera():
#             print("❌ Camera setup failed, aborting camera test.")
#             print("=== sanitytest finished (camera setup failed) ===\n")
#         else:
#             print("✅ Camera setup OK.")

#             # 3) Grab one frame and run full pipeline
#             print("[3] Grabbing one frame via take_image() …")
#             try:
#                 trig.take_image(cam_idx=0, sr=None)  # SR model optional for test
#                 img = trig.capture_image()
#                 if img is None:
#                     print("⚠️ No frame found in queue.")
#                 else:
#                     print(f"✅ Captured preview frame: shape={img.shape}")

#                     # 4) Try detection + save (will catch YOLO errors)
#                     print("[4] Running one-shot detection + save …")
#                     try:
#                         trig._run_one_shot_detection(img, cam_idx=0)
#                     except Exception as e:
#                         print(f"⚠️ Detection failed during sanitytest (this is OK if YOLO not configured): {e}")

#             except Exception as e:
#                 print(f"⚠️ Error during camera grab test: {e}")
#             finally:
#                 print("[5] Stopping camera + PLC …")
#                 trig.stop()

#             print("=== sanitytest finished (camera path) ===\n")



