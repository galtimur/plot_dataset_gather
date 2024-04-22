import json
from pathlib import Path
from typing import Dict, List

import nbformat as nbf


def save_jsonl(data, file_path: str | Path) -> None:
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def read_jsonl(file_path: str | Path) -> List[Dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_nb_data_cell(nb_path: str | Path):
    with open(nb_path) as f:
        nb = nbf.read(f, as_version=4)

    data_code = nb.cells[0]["source"]
    return data_code


def read_responses(file_path: str | Path) -> dict:
    responses = read_jsonl(file_path)

    responses_dict = dict()

    for entry in responses:
        if "id" in entry:
            responses_dict[entry["id"]] = entry

    return responses_dict
