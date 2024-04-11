import os
import json
from pathlib import Path
import pandas as pd
import nbformat as nbf

def get_dp_folders(folder_path):
    subfolder_list = []
    dp_folders = sorted(os.listdir(folder_path))

    for name in dp_folders:

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

def get_pycharm_dataframe_description(df: pd.DataFrame) -> str:
    descr_lines = [f'Number of rows in DataFrame: {len(df)}']
    descr_lines.append('DataFrame has the following columns:')
    for col in df.columns:
        types_set = set(df.loc[df[col].notna(), col].apply(type))
        types_list = [str(type_.__name__) for type_ in types_set]
        if len(types_list) == 1:
            col_types = types_list.pop()
        else:
            col_types = str(set(types_list)).replace('"', '').replace('\'', '')
        descr = f'{col} of type {col_types}. Count: {df[col].count()}'
        if str(df[col].dtype).startswith(('int', 'float')):
            mean = f'{df[col].mean():.6}'
            std = f'{df[col].std():.6}'
            if str(df[col].dtype).startswith('int'):
                minimum = f'{df[col].min()}'
                maximum = f'{df[col].max()}'
            else:
                minimum = f'{df[col].min():.6}'
                maximum = f'{df[col].max():.6}'
            descr = descr + f', Mean: {mean}, Std. Deviation: {std}, Min: {minimum}, Max: {maximum}'
        descr_lines.append(descr)
    return '\n'.join(descr_lines)

def read_nb_data_cell(nb_path):

    with open(nb_path) as f:
        nb = nbf.read(f, as_version=4)

    data_code = nb.cells[0]['source']
    return data_code
