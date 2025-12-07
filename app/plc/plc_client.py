# TODO: implement
# app/plc/plc_client.py
from typing import Optional

try:
    from PyXGT.LS import plc_ls
except Exception:
    plc_ls = None

class PLCClient:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.conn = None

    def connect(self) -> bool:
        if plc_ls is None:
            print("⚠️ PLC driver unavailable (PyXGT.LS import failed).")
            return False
        try:
            self.conn = plc_ls(self.ip, self.port)
            return True
        except Exception as e:
            print(f"⚠️ PLC connect error: {e}")
            self.conn = None
            return False

    def write_bit(self, addr: str, val: str) -> None:
        if not self.conn:
            return
        try:
            self.conn.command("XGB", "write", "bit", addr, val)
        except Exception as e:
            print(f"⚠️ PLC write_bit error: {e}")

    def read_bit(self, addr: str) -> Optional[int]:
        if not self.conn:
            return None
        try:
            raw = self.conn.command("XGB", "read", "bit", addr)
            if isinstance(raw, (list, tuple)) and raw:
                return int(raw[0])
            return int(raw)
        except Exception as e:
            print(f"⚠️ PLC read_bit error: {e}")
            return None

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
# if __name__ == "__main__":
#     from app.core.config import settings

#     plc = PLCClient(ip=settings.PLC_IP, port=settings.PLC_PORT)
#     ok = plc.connect()
#     print("Connected:", ok)
#     if ok:
#         val = plc.read_bit("D7100")
#         print("D7100:", val)
#         plc.write_bit("D7020", 1)
#         plc.write_bit("D7020", 0)
#         plc.close()
#     else:
#         print("No PLC.")
