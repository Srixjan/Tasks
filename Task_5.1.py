import os
def write_log_entry(filepath: str, message: str) -> None:
    with open(filepath, "a") as f:
        f.write(message + "\n")

def read_log_entries(filepath: str) -> list:
    log_entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            log_entries.append(line.rstrip("\n\r"))
    return log_entries

def delete_log_file(filepath: str) -> bool: 
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    else:
        return False

def main():
    log_path = "pipeline.log"
    write_entries = "Batch 12 processed successfully"
    write_log_entry(log_path, write_entries)

    read_logs = read_log_entries(log_path)
    print(read_logs)

    delete_log_file(log_path)

if __name__ == "__main__":
    main()