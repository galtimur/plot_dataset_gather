import json

import nbformat as nbf


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


def read_nb_data_cell(nb_path):
    with open(nb_path) as f:
        nb = nbf.read(f, as_version=4)

    data_code = nb.cells[0]["source"]
    return data_code
