from pathlib import Path
from omegaconf import OmegaConf
import nbformat as nbf
import subprocess

from utils import read_jsonl

def filter_prints(code: str):

    '''
    remove lines with prints
    '''

    lines = code.split("\n")
    lines_cleaned = [line for line in lines if not line.startswith("print(")]
    code_cleaned = "\n".join(lines_cleaned)

    return code_cleaned

def get_code_blocks(gpt_response: dict):

    '''
    parse codeblocks (highlighted by ```) from LLM response
    '''

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

def build_new_nb(blocks: list, nb_path):

    '''
    save codeblocks into notebook
    '''

    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(block) for block in blocks]

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    dataset_folder = Path(config.matplotlib_dataset_path)
    code_split_results = read_jsonl(dataset_folder / "gpt_response.jsonl")
    # first line is model prompt
    code_split_results = code_split_results[1:]

    for gpt_response in code_split_results:
        idx = gpt_response["id"]
        dp_folder = dataset_folder / str(idx)
        nb_path = dp_folder / 'split_data.ipynb'
        code_blocks = get_code_blocks(gpt_response)

        # we assume that the code block is first, but just in case, I formulate it is penultimate
        data_block = code_blocks[-2]
        data_block += "\ndf.to_csv('data.csv', index=False)"
        data_block_file = dp_folder / 'data_block.py'
        with open(data_block_file, 'w') as f:
            f.write(data_block)
        # run data script to generate the data file
        subprocess.run(['python', data_block_file], cwd=dp_folder)

        # add printing df into notebook
        code_blocks[-2] += "\ndf.head(15)"

        # add a block with full plotting script to be able to compare results and code
        code_file = dp_folder / "plot.py"
        with open(code_file, "r") as f:
            code_joined = f.read()
        code_blocks.append(code_joined)

        build_new_nb(code_blocks, nb_path)

        # run the notebook to generate all outputs
        cmd = f'jupyter nbconvert --execute --to notebook --inplace "{nb_path}"'
        subprocess.call(cmd, shell=True)

