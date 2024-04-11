from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import json
import os
import random
from dataclasses import dataclass

from utils import get_dp_folders
from GPT4V_backbone import GPT4V

@dataclass
class PipelineParameters:

    dataset_folder: Path
    output_file: Path
    openai_token: str
    instructs: dict


def prepare_pipeline(config_path, out_filename):

    random.seed(42)

    config = OmegaConf.load(config_path)
    openai_token_file = config.openai_token_file
    dataset_folder = Path(config.dataset_final)
    out_folder = Path(config.out_folder)
    output_file = out_folder / out_filename

    os.makedirs(out_folder, exist_ok=True)

    with open(openai_token_file, "r") as f:
        openai_token = f.read()

    with open("prompts/plot_gen.json", 'r') as f:
        instructs = f.read()
        instructs = json.loads(instructs)

    return PipelineParameters(dataset_folder, output_file, openai_token, instructs)


def generate_plotting_request(dp_folder: Path, instructs):

    "Request to ask model to write a code for plotting. Add dataframe description"

    df_descr_file = dp_folder / "data_descr.txt"
    task_file = dp_folder / "task.json"

    with open(task_file, 'r') as f:
        task_dict = json.load(f)

    with open(df_descr_file, 'r') as f:
        df_descr = f.read()

    task = instructs["plot instruct"]
    for i, (task_type, task_part) in enumerate(task_dict.items(), start=1):
        task += f"{i}.{task_part}\n"
        if task_type == "data description":
            task += instructs["data instruct"] + df_descr + "\n"

    task = task.replace("\n\n", "\n")

    return task

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    pipline_parameters = prepare_pipeline(config_path,"gpt_plot.jsonl")

    with open(pipline_parameters.output_file, "a") as f:
        json.dump(pipline_parameters.instructs, f)
        f.write("\n")

    gpt4v = GPT4V(api_key=pipline_parameters.openai_token, system_prompt=pipline_parameters.instructs["system prompt"])
    responses = []

    dp_folders = get_dp_folders(pipline_parameters.dataset_folder)
    dp_folders = random.sample(dp_folders, 2)
    for i, dp_folder in tqdm(enumerate(dp_folders), total=len(dp_folders)):

        index = int(dp_folder.name)
        task = generate_plotting_request(dp_folder, pipline_parameters.instructs)
        response = gpt4v.make_request(task)

        if response is None:
            print(f"Skipping dp {index}")
            continue

        response["id"] = index
        response["task"] = task
        responses.append(response)

        with open(pipline_parameters.output_file, "a") as f:
            json.dump(response, f)
            f.write("\n")
