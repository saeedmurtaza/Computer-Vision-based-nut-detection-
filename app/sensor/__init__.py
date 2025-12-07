# app/sensor/__init__.py
from .serial_sensor import sensor_init, read_proximity_json

__all__ = ["sensor_init", "read_proximity_json"]
