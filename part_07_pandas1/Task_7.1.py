import pandas as pd
def build_drone_series(drone_id: str, battery_pct: int, altitude_m: float, speed_kmh: float, status: str) ->pd.Series:
    reading_data = {"drone_id": drone_id,
                "battery_pct": battery_pct,
                "altitude_m": altitude_m,
                "speed_kmh": speed_kmh,
                "status": status}
    
    reading = pd.Series(data=reading_data, name=drone_id)
    return reading

reading = build_drone_series(drone_id="D101", 
                             battery_pct=76, 
                             altitude_m=120.5, 
                             speed_kmh=42.3, 
                             status="active")

print(reading.name)

readings_list = [
     {"drone_id": "D101", "battery_pct": 76, "altitude_m": 120.5, "speed_kmh": 42.3, "status": "active"},
     {"drone_id": "D102", "battery_pct": 40, "altitude_m": 0.0, "speed_kmh": 0.0, "status": "charging"},
     {"drone_id": "D103", "battery_pct": 88, "altitude_m": 95.2, "speed_kmh": 30.1, "status": "idle"},
     {"drone_id": "D104", "battery_pct": 55, "altitude_m": 150.0, "speed_kmh": 48.7, "status": "active"},
     {"drone_id": "D105", "battery_pct": 12, "altitude_m": 0.0, "speed_kmh": 0.0, "status": "charging"}]

def build_fleet_dataframe(readings: list[dict]) -> pd.DataFrame:
    dataframe = pd.DataFrame(data=readings)
    return dataframe

df = build_fleet_dataframe(readings_list)
print(df)
print(df.shape)
print(df.dtypes)

def inspect_dataframe(df: pd.DataFrame) -> dict:
    view_dict = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype)for col, dtype in df.dtypes.items()},
        "has_object_columns": any(dtype == "object" for dtype in df.dtypes)
    }
    return view_dict

print(inspect_dataframe(df))