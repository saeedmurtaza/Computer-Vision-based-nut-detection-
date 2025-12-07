# app/camera/basler_driver.py
from typing import Any, Optional

from pypylon import pylon, genicam


class BaslerCamera:
    """
    Thin convenience wrapper over pypylon.InstantCamera.

    Used by ProximityCameraTrigger All low-level grab logic
    is centralized here.
    """

    def __init__(self, device_info: Any, name: str = "Basler"):
        self.device_info = device_info
        self.name = name
        self.camera: Optional[pylon.InstantCamera] = None
        self.converter: Optional[pylon.ImageFormatConverter] = None
    print(f"---------------------------checking basler camera---------------------------")
    def open(self) -> bool:
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            ip_device = tl_factory.CreateDevice(self.device_info)
            self.camera = pylon.InstantCamera(ip_device)
            self.camera.Open()

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            return True
        except genicam.GenericException as e:
            print(f"❌ BaslerCamera.open() failed for {self.name}: {e}")
            self.camera = None
            self.converter = None
            return False

    def is_open(self) -> bool:
        return bool(self.camera and self.camera.IsOpen())

    def set_param(self, key: str, value):
        if not self.is_open():
            return
        try:
            nm = self.camera.GetNodeMap()
            node = nm.GetNode(key)
            if node and genicam.IsWritable(node):
                try:
                    node.SetValue(float(value) if isinstance(value, (int, float)) else value)
                except Exception:
                    node.FromString(str(value))
        except Exception as e:
            print(f"⚠️ BaslerCamera.set_param({key}={value}) failed: {e}")

    def start_single_grab(self):
        if self.camera:
            self.camera.StartGrabbing(1)

    def retrieve_one(self, timeout_ms: int = 5000):
        """
        Returns (success: bool, bgr_image: np.ndarray or None).
        """
        if not self.camera or not self.converter:
            return False, None
        try:
            gr = self.camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            if not gr.GrabSucceeded():
                gr.Release()
                return False, None
            image = self.converter.Convert(gr)
            gr.Release()
            return True, image.GetArray()
        except pylon.TimeoutException:
            return False, None
        except Exception as e:
            print(f"⚠️ BaslerCamera.retrieve_one() error: {e}")
            return False, None

    def stop_grab(self):
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except Exception:
            pass

    def close(self):
        try:
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
        except Exception:
            pass
        self.camera = None
        self.converter = None

#  Test block for BaslerCamera class

# if __name__ == "__main__":
#     print("-" * 50)
#     print("BASLER CAMERA DRIVER TEST INITIATED")
#     print("-" * 50)
    
#     # 1. DISCOVERY: Find all connected cameras
#     try:
#         tl_factory = pylon.TlFactory.GetInstance()
#         devices = tl_factory.EnumerateDevices()
#         if not devices:
#             print("❌ No Basler cameras found. Test aborted.")
#             exit()
        
#         # We will test with the first camera found
#         device_info = devices[0]
#         print(f"✅ Found camera: {device_info.GetModelName()} ({device_info.GetSerialNumber()})")
        
#         # 2. INITIALIZATION
#         camera_name = f"TestCam-{device_info.GetSerialNumber()}"
#         cam = BaslerCamera(device_info, name=camera_name)
        
#         # 3. OPEN AND CONFIGURE
#         print(f"-> Attempting to open {cam.name}...")
#         if not cam.open():
#             print("❌ Failed to open camera. Exiting test.")
#             exit()
            
#         print(f"-> Camera opened successfully. Testing methods...")
        
#         # Test 3a: set_param (Example: setting a short exposure time)
#         cam.set_param("ExposureTime", 10000) # Set exposure to 10,000 microseconds
#         print(f"-> Set ExposureTime to 10000. Is camera open? {cam.is_open()}")
        
#         # 3b: start_single_grab
#         print("-> Starting single frame grab...")
#         cam.start_single_grab()
        
#         # 3c: retrieve_one
#         print("-> Attempting to retrieve image...")
#         success, image_array = cam.retrieve_one(timeout_ms=3000) # 3-second timeout
        
#         if success:
#             print(f"✅ Image retrieved successfully! Shape: {image_array.shape}, Data Type: {image_array.dtype}")
#         else:
#             print("❌ Image retrieval failed (check lighting/exposure).")

#     except Exception as e:
#         print(f"\nFATAL TEST ERROR: {e}")
        
#     finally:
#         # 4. CLEANUP: Ensure the camera is closed regardless of errors
#         if 'cam' in locals() and cam.is_open():
#             print(f"-> Stopping grab and closing camera...")
#             cam.stop_grab()
#             cam.close()
#             print("✅ Cleanup complete. Test finished.")
#         else:
#             print("Test finished without a camera cleanup required.")