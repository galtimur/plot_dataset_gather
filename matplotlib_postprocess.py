from pathlib import Path
from omegaconf import OmegaConf
import nbformat as nbf

from utils import read_jsonl

def filter_prints(code):
    lines = code.split("\n")
    lines_cleaned = [line for line in lines if not line.startswith("print(")]
    code_cleaned = "\n".join(lines_cleaned)

    return code_cleaned

def get_code_blocks(gpt_response: dict):
    code_start_seq = "python\n"
    cut = len(code_start_seq)
    code = gpt_response["prediction"]
    response_blocks = code.split("```")

    code_blocks = []
    for block in response_blocks[1::2]:
        if block.startswith(code_start_seq):
            block = block[cut:]
        code_blocks.append(filter_prints(block))

    return code_blocks

def build_new_nb(blocks, nb_path):

    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(block) for block in blocks]

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    dataset_folder = Path(config.matplotlib_dataset_path)
    code_split_results = read_jsonl(dataset_folder / "gpt_response.jsonl")

    for gpt_response in code_split_results:
        idx = gpt_response["id"]
        nb_path = f'out/test_notebook_{idx}.ipynb'
        code_blocks = get_code_blocks(gpt_response)
        build_new_nb(code_blocks, nb_path)

# %%