from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import json
import glob
import os
import random

from utils import get_dp_folders
from GPT4V_backbone import GPT4V

def generate_task_request(code: str, df_summary: str, instructs: dict):

    code_text = f"CODE:\n{code}"
    df_text = f"Dataframe SUMMARY:\n{df_summary}"

    request = [instructs["part 1"], code_text, df_text, instructs["part 2"]]
    request = "\n".join(request)

    return request

if __name__ == "__main__":

    random.seed(42)

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    openai_token_file = config.openai_token_file
    dataset_folder = config.dataset_valid_step_1

    output_file = Path(dataset_folder) / "gpt_tasks.jsonl"

    existing_ids = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            for line in file:
                json_line = json.loads(line)
                if 'id' in json_line:  # to ensure that the key exists
                    existing_ids.append(json_line['id'])

    with open(openai_token_file, "r") as f:
        openai_token = f.read()

    with open("prompts/task_gen.json", 'r') as f:
        instructs = f.read()
        instructs = json.loads(instructs)

    gpt4v = GPT4V(api_key=openai_token, system_prompt=instructs["system prompt"])

    if not os.path.exists(output_file):
        with open(output_file, "a") as f:
            json.dump(instructs, f)
            f.write("\n")

    dp_folders = get_dp_folders(dataset_folder)
    responses = []
    # dp_folders = dp_folders[:1]
    # dp_folders = random.sample(dp_folders, 20)
    for i, dp_folder in tqdm(enumerate(dp_folders), total=len(dp_folders)):

        index = int(dp_folder.name)
        if index in existing_ids:
            continue

        code_file = dp_folder / "plot.py"
        df_sum_file = dp_folder / "data_descr.txt"
        plot_files = glob.glob(os.path.join(str(dp_folder), "*.png"))

        with open(code_file, "r") as f:
            code = f.read()

        with open(df_sum_file, "r") as f:
            df_summary = f.read()

        request = generate_task_request(code, df_summary, instructs)
        response = gpt4v.make_request(request=request, image_paths=plot_files, image_detail="low")

        if response is None:
            print(f"Skipping dp {index}")
        else:
            response["id"] = index
            responses.append(response)
            with open(output_file, "a") as f:
                json.dump(response, f)
                f.write("\n")
