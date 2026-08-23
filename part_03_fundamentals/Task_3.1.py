def parse_sensor_id(raw: str) -> tuple[str, str]:
    raw = raw.split("-")
    t1 = (raw[1])
    t2 = (raw[2]).upper()

    return t1, t2

def format_label(rawi: str) -> str:
    rawi = parse_sensor_id(rawi)
    return f"Sensor #{rawi[0]} ({rawi[1]})"

psid = input("Enter the Parse Sensor ID: ").strip()
print(format_label(psid))