import glob
import json
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from LLM_utils import read_task_responses
from utils import read_nb_data_cell

"""
Gather datapoints
"""


def format_task(message):
    """
    Splits the list of items like 1. \n 2. \n ... and forms a dict
    """

    parse_errs_file = "out/parsing_errors.txt"

    # parts = re.findall(r'\d+\.\s*(.*)', message)
    message = " \n" + message

    part = message
    parts = []
    for i in range(1, 5):
        split = part.split(f"\n{i}.")
        parts.append(split[0])
        part = split[1]
    parts.append(part)
    parts.pop(0)

    task_dict = {
        "setup": parts[0],
        "data description": parts[1],
        "plot description": parts[2],
        "plot style": parts[3],
    }

    return task_dict


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    openai_token_file = config.openai_token_file

    dataset_folder = Path(config.dataset_valid_step_1)
    output_folder = Path(config.out_folder)
    dataset_folder_final = Path(config.dataset_final)
    response_path = output_folder / "gpt_tasks.jsonl"
    os.makedirs(dataset_folder_final, exist_ok=True)

    response = read_task_responses(response_path)
    dp_ids = sorted(list(response.keys()))

    files_list = ["plot.py", "data_descr.txt", "data.csv", "plot_original.py"]

    for idx in tqdm(dp_ids):
        task_dict = format_task(response[idx])
        dp_folder = dataset_folder / str(idx)
        dp_folder_final = dataset_folder_final / str(idx)
        os.makedirs(dp_folder_final, exist_ok=True)

        dp_files = glob.glob(os.path.join(str(dp_folder), "*.png"))
        dp_files = [Path(file) for file in dp_files]

        for file in files_list:
            plot_code_file = dp_folder / file
            dp_files.append(plot_code_file)

        data_code = read_nb_data_cell(dp_folder / "split_data_cut.ipynb")
        data_code_file = dp_folder_final / "data_load.py"
        task_file = dp_folder_final / "task.json"

        with open(data_code_file, "w") as f:
            f.write(data_code)

        with open(task_file, "w") as f:
            json.dump(task_dict, f)

        for file in dp_files:
            shutil.copy2(file, dp_folder_final / file.name)
