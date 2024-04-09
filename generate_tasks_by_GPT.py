from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import json
import glob
import os
import time
import re

from utils import get_dp_folders
from GPT4V_backbone import GPT4V

config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
openai_token_file = config.openai_token_file
dataset_folder = config.dataset_valid_step_1

output_file = Path(dataset_folder) / "gpt_tasks_detailed.jsonl"

existing_ids = []
if os.path.exists(output_file):
    with open(output_file, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            if 'id' in json_line:  # to ensure that the key exists
                existing_ids.append(json_line['id'])

with open(openai_token_file, "r") as f:
    openai_token = f.read()

instruction_1 = "Write the detailed TASK to write a code for plotting the given pandas dataframe. Code and dataframe summary is given below. Result of the generated plot image(s) is given below.\n"
instruction_2 = "\nWrite the detailed TASK to write a CODE for plotting the given pandas dataframe (SUMMARY given), to produce plot image (image given at the end). Do not gemerate dataframe, imply that it is given. Write only task after word TASK."

def generate_request(code: str, df_summary: str):

    code_text = f"CODE:\n{code}"
    df_text = f"Dataframe SUMMARY:\n{df_summary}"

    request = [instruction_1, code_text, df_text, instruction_2]
    request = "\n".join(request)

    return request

system_prompt = "You are a helpful programming assistant proficient in python, matplotlib and pandas dataframes."
gpt4v = GPT4V(api_key=openai_token, system_prompt=system_prompt)

prompts = {"system prompt": system_prompt, "instruction_1": instruction_1, "instruction_2": instruction_2}

if not os.path.exists(output_file):
    with open(output_file, "a") as f:
        json.dump(prompts, f)
        f.write("\n")

dp_folders = get_dp_folders(dataset_folder)
responses = []
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

    request = generate_request(code, df_summary)
    do_request = True

    error_counts = 0
    while do_request and error_counts<10:
        response = gpt4v.ask(request=request, image_paths=plot_files, image_detail="low")

        if "error" not in response.keys():

            response["id"] = index
            responses.append(response)

            with open(output_file, "a") as f:
                json.dump(response, f)
                f.write("\n")
            do_request = False
        else:
            message = response["error"]["message"]
            seconds_to_wait = re.search(r'Please try again in (\d+)s\.', message)
            if seconds_to_wait is not None:
                wait_time = 1.5*int(seconds_to_wait.group(1))
                print(f"Waiting {wait_time} s")
                time.sleep(wait_time)
            else:
                print(f"Cannot parse retry time from error message. Skiping dp {index}")
                print(message)
                time.sleep(20)
                error_counts += 1
