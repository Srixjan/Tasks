import os
def write_log_entry(filepath: str, message: str) -> None:
    with open(filepath, "a") as f:
        f.write(message + "\n")

def read_log_entries(filepath: str) -> None:
    log_entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            log_entries.append(line.rstrip("\n\r"))
    return log_entries

def delete_log_file(filepath: str) -> None: 


write_log_entry("pipeline.log", "Batch 12 processed successfully")
write_log_entry("pipeline.log", "Batch 13 failed: missing sensor_id")
print(read_log_entries("pipeline.log"))