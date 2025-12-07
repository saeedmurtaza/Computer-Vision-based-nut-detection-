# TODO: implement
# app/sensor/serial_sensor.py
import serial
import json
import time
from typing import Tuple, Optional

PORT = None #'COM6' since arduino is not connected to test we set it to None
BAUD = 115200
TIMEOUT = 1

def sensor_init() -> serial.Serial:
    """
    Open the Arduino-based proximity sensor on COM6.

    Returns:
        serial.Serial object
    """
    if not PORT:
        raise ValueError("❌ Proximity sensor PORT not specified.")
    else:
        print(f"---------------------------checking proximity sensor---------------------------")
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        # small delay to let Arduino reset
        time.sleep(1.0)
        print(f"✅ Connected to Arduino on {PORT} @ {BAUD} baud")
        return ser


def read_proximity_json(ser: serial.Serial) -> Optional[Tuple[int, int, int]]:
    """
    Read one JSON line from Arduino and parse it.

    Expected JSON format:
        {"detected": 0/1, "count": int, "analog": int}

    Returns:
        (detected, count, analog) or None if line invalid.
    """
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return None
        data = json.loads(line)
        detected = int(data.get("detected", 1))
        count = int(data.get("count", 0))
        analog = int(data.get("analog", 0))
        return detected, count, analog
    except Exception as e:
        # You can add a logger here later
        print(f"⚠️ Proximity JSON parse error: {e}")
        return None

#  # Test block for serial sensor functions
if __name__ == "__main__":
    # 1. Initialize the serial connection
    try:
        ser_conn = sensor_init()
    except ValueError as e:
        print(e)
        exit()

    # 2. Test reading data 10 times
    try:
        print("\n--- Starting Proximity Sensor Read Test ---")
        for i in range(1, 11):
            # Call the function correctly, passing the established serial connection object
            data_tuple = read_proximity_json(ser_conn)
            
            if data_tuple:
                detected, count, analog = data_tuple
                print(f"[{i}/10] Status: Detected={detected}, Count={count}, Analog={analog}")
            else:
                print(f"[{i}/10] Warning: Failed to parse line or received empty data.")
            
            # Add a small delay for stable output viewing
            time.sleep(0.2) 

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    finally:
        # 3. Close the serial connection
        if 'ser_conn' in locals() and ser_conn.is_open:
            ser_conn.close()
            print(f"\n✅ Serial connection on {PORT} closed.")
