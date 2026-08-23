class RecordValidatorError(Exception):
    pass

class MissingFieldError(RecordValidatorError):
    pass

class InvalidReadingError(RecordValidatorError):
    pass

def validate_record(record: dict) -> bool:
    sensor_id = record.get("sensor_id")
    if not sensor_id:
        raise MissingFieldError("sensor_id is missing or empty")

    timestamp_id = record.get("timestamp")
    if not timestamp_id:
        raise MissingFieldError("timestamp_id is missing or empty")

    reading = record.get("reading")
    if reading is None:
        raise InvalidReadingError("reading is missing!")
    if isinstance(reading, bool):
        raise InvalidReadingError("reading is boolean!")
    if not isinstance(reading, (int, float)):
        raise InvalidReadingError(f"reading must be a number, got{type(reading).__name__}")
     
    return True

def process_batch(records: list[dict]) -> dict:
    processed = 0
    skipped = 0
    errors = []

    for i, record in enumerate(records):
        identifier = record.get("sensor_id") or f"index_{i}"
        try:
            validate_record(record)
            processed += 1
        except RecordValidatorError as e:
            skipped += 1
            errors.append(f"{identifier}: {e}")

    return {"processed": processed, "skipped": skipped, "errors": errors}

records = [
    {"sensor_id": "S1", "reading": 23.5, "timestamp": "2025-01-01"},
    {"sensor_id": "", "reading": 23.5, "timestamp": "2025-01-01"},
]
process_batch(records)