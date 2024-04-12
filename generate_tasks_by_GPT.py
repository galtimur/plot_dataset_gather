from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import json
import glob
import os
import random

from utils import get_dp_folders
from GPT4V_backbone import GPT4V
from LLM_utils import generate_task_request, prepare_pipeline


if __name__ == "__main__":

    # TODO use prepare_pipeline function instead

    config_path = "configs/config.yaml"
    out_filename = "gpt_tasks.jsonl"
    pipline_parameters = prepare_pipeline(config_path, out_filename, "prompts/task_gen.json")

    gpt4v = GPT4V(api_key=pipline_parameters.openai_token, system_prompt=pipline_parameters.instructs["system prompt"])

    dp_folders = get_dp_folders(pipline_parameters.dataset_folder)
    responses = []
    # dp_folders = dp_folders[:1]
    # dp_folders = random.sample(dp_folders, 20)
    for i, dp_folder in tqdm(enumerate(dp_folders), total=len(dp_folders)):

        index = int(dp_folder.name)
        if index in pipline_parameters.existing_ids:
            continue

        code_file = dp_folder / "plot.py"
        df_sum_file = dp_folder / "data_descr.txt"
        plot_files = glob.glob(os.path.join(str(dp_folder), "*.png"))
        plot_files = [Path(file) for file in plot_files]

        with open(code_file, "r") as f:
            code = f.read()

        with open(df_sum_file, "r") as f:
            df_summary = f.read()

        request = generate_task_request(code, df_summary, pipline_parameters.instructs)
        response = gpt4v.make_request(request=request, images=plot_files, image_detail="low")

        if response is None:
            print(f"Skipping dp {index}")
        else:
            response["id"] = index
            responses.append(response)
            with open(pipline_parameters.output_file, "a") as f:
                json.dump(response, f)
                f.write("\n")
