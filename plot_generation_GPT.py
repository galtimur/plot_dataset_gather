from pathlib import Path
from tqdm import tqdm
import json

from utils import get_dp_folders
from GPT4V_backbone import GPT4V
from LLM_utils import prepare_pipeline, generate_plotting_request




if __name__ == "__main__":

    config_path = "configs/config.yaml"
    out_filename = "gpt_plots.jsonl"
    pipline_parameters = prepare_pipeline(config_path, out_filename, "prompts/plot_gen.json")

    with open(pipline_parameters.output_file, "a") as f:
        json.dump(pipline_parameters.instructs, f)
        f.write("\n")

    gpt4v = GPT4V(api_key=pipline_parameters.openai_token, system_prompt=pipline_parameters.instructs["system prompt"])
    responses = []

    dp_folders = get_dp_folders(pipline_parameters.dataset_folder)
    # dp_folders = random.sample(dp_folders, 2)
    for i, dp_folder in tqdm(enumerate(dp_folders), total=len(dp_folders)):

        index = int(dp_folder.name)

        if index in pipline_parameters.existing_ids:
            continue

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
