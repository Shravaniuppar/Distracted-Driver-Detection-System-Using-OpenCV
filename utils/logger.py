# utils/logger.py
import csv
import os
from datetime import datetime

LOG_FILE = "driver_status_log.csv"
if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "status"])
def log_status(status):
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "status"])  # Write header once
        writer.writerow([datetime.now().isoformat(), status])
