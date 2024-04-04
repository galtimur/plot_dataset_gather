import os
import json
from pathlib import Path

def get_dp_folders(folder_path):
    subfolder_list = []

    for name in os.listdir(folder_path):

        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            subfolder_list.append(Path(full_path))

    return subfolder_list

def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
