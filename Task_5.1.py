def write_log_entry(filepath: str, message: str) -> None:
    with open(filepath, "a") as f:
        f.write(message + "\n")


write_log_entry("pipeline.log", "Batch 12 processed successfully")
write_log_entry("pipeline.log", "Batch 13 failed: missing sensor_id")