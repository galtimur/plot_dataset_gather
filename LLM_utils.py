import random
from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import os
import json
import glob
import tiktoken

@dataclass
class PipelineParameters:

    config: DictConfig
    dataset_folder: Path
    output_file: Path
    out_folder: Path
    openai_token: str
    instructs: dict
    existing_ids: list


def prepare_pipeline(config_path, out_filename, prompt_file_path):

    random.seed(42)

    config = OmegaConf.load(config_path)
    openai_token_file = config.openai_token_file
    dataset_folder = Path(config.dataset_final)
    out_folder = Path(config.out_folder)
    output_file = out_folder / out_filename

    os.makedirs(out_folder, exist_ok=True)

    with open(openai_token_file, "r") as f:
        openai_token = f.read()

    with open(prompt_file_path, 'r') as f:
        instructs = f.read()
        instructs = json.loads(instructs)

    existing_ids = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            for line in file:
                json_line = json.loads(line)
                if 'id' in json_line:  # to ensure that the key exists
                    existing_ids.append(json_line['id'])

    return PipelineParameters(config, dataset_folder, output_file, out_folder, openai_token, instructs, existing_ids)

def read_task_responses(response_file):
    response_dict = {}
    with open(response_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            entry_id = entry.get('id')
            if entry_id is not None:
                message = entry['choices'][0]['message']['content']
                if message:
                    if message.startswith("TASK:"):
                        message = message[5:].lstrip('\n ').replace("**", "")
                    response_dict[entry_id] = message
                if entry["probs"]:
                    logprobs = entry['choices'][0]['logprobs']['content'][0]['top_logprobs']
                    response_dict[entry_id] = {"message": message, "logprobs": logprobs}

    return response_dict

def generate_task_request(code: str, df_summary: str, instructs: dict):

    '''
    Build request to generate a task for the model
    '''

    code_text = f"CODE:\n{code}"
    df_text = f"Dataframe SUMMARY:\n{df_summary}"

    request = [instructs["part 1"], code_text, df_text, instructs["part 2"]]
    request = "\n".join(request)

    return request


def generate_benchmark_request(dp_folder: Path, instructs: dict, result: dict):

    "Request to ask model to write a code for plotting. Add dataframe description"

    # df_descr_file = dp_folder / "data_descr.txt"
    plot_files = glob.glob(os.path.join(str(dp_folder), "*.png"))
    plot_file_gt = Path(plot_files[0])
    plot_gen = result["images"][0]
    task = instructs["request judge"]
    plots = [plot_gen, plot_file_gt]

    return task, plots

def construct_logit_args(options, model_name="gpt-4-turbo"):

    tokenizer = tiktoken.encoding_for_model(model_name)

    options = [str(i) for i in list(range(0, 11))]

    options_tok_ids = dict()
    logit_bias = dict()

    for opt in options:
        tok_ids = tokenizer.encode(opt)
        assert len(tok_ids) == 1
        logit_bias[tok_ids[0]] = 30
        options_tok_ids[opt] = tok_ids

    args = {"max_tokens": 1, "temperature": 0.3, "n": 1, "logprobs": True, "top_logprobs": 20, "logit_bias": logit_bias}