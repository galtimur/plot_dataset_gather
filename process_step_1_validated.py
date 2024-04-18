import os
import shutil
from tqdm import tqdm
import subprocess
import nbformat as nbf
import glob
from pathlib import Path
from omegaconf import OmegaConf
import base64
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from data import get_dp_folders
from user_api import get_pycharm_dataframe_description


def copy_valid_dp(data_folder, source_folder, target_folder):
    list_of_files = glob.glob(f'{str(source_folder)}\\*.ipynb')

    os.makedirs(target_folder, exist_ok=True)
    list_of_ids = [int(f.split('_')[-1].split('.')[0]) for f in list_of_files]

    for idx, file in tqdm(zip(list_of_ids, list_of_files)):

        dp_folder = data_folder / str(idx)
        target_dp_folder = target_folder / str(idx)
        nb_file = target_dp_folder / "split_data.ipynb"

        if not os.path.exists(target_dp_folder):
            shutil.copytree(dp_folder, target_dp_folder)

        shutil.copy2(file, nb_file)

        if os.path.exists(target_dp_folder / "plot.py"):
            os.rename(target_dp_folder / "plot.py", target_dp_folder / "plot_original.py")

def cut_noteboook(notebook_path):
    '''
    Keep only second cell in the notebook
    '''

    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    nb.cells = nb.cells[1:2]

    nb['cells'][0]['outputs'] = []
    nb.cells[0]['source'] = 'import pandas as pd\n' + 'df = pd.read_csv("data.csv")\n' + nb.cells[0]['source']

    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)


def clean_noteboook(notebook_path):
    '''
    Keep only first cell in the notebook
    '''

    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    nb.cells = nb.cells[:1]

    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)


def generate_stand_alone_dps(folder):
    '''
    generates stand-alone notebook (only loading data from csv and plotting) and runs it
    '''

    dp_folders = get_dp_folders(folder)

    for dp_folder in tqdm(dp_folders):
        nb_path_orig = dp_folder / "split_data.ipynb"
        nb_path_copy = dp_folder / "split_data_cut.ipynb"

        shutil.copy2(nb_path_orig, nb_path_copy)

        cut_noteboook(nb_path_copy)

        cmd = f'jupyter nbconvert --execute --to notebook --inplace "{nb_path_copy}"'
        subprocess.call(cmd, shell=True)


def gather_nbs(folder, target_folder):
    '''
    gather notebooks from datapoint folders to single folder
    '''

    os.makedirs(target_folder, exist_ok=True)

    dp_folders = get_dp_folders(folder)

    for dp_folder in tqdm(dp_folders):
        idx = int(dp_folder.name)
        nb_path_cut = dp_folder / "split_data_cut.ipynb"
        nb_path_copy = target_folder / f"split_data_cut_{idx}.ipynb"

        shutil.copy2(nb_path_cut, nb_path_copy)


def clean_dp_nbs(folder):
    '''
    keep only one code cell in the notebook
    '''

    dp_folders = get_dp_folders(folder)

    for dp_folder in tqdm(dp_folders):
        nb_path_copy = dp_folder / "split_data_cut.ipynb"
        clean_noteboook(nb_path_copy)


def gather_plot_from_nb(nb_path):
    '''
    Extracts plot from the output of the first cell of the notebook
    '''

    dp_folder = nb_path.parent
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    outputs = nb.cells[1]['outputs']
    if len(outputs) == 0:
        print(dp_folder.name)
        return None

    for i, output in enumerate(outputs):
        if output['output_type'] == 'display_data' and 'image/png' in output['data']:
            image_data = output['data']['image/png']

            if len(outputs) == 1:
                suffix = ""
            else:
                suffix = f"_{i}"

            with open(dp_folder / f'plot{suffix}.png', 'wb') as f:
                f.write(base64.b64decode(image_data))


def gather_plots(folder):
    '''
    Extracts plots from notebooks in all datapoints
    '''

    dp_folders = get_dp_folders(folder)

    for dp_folder in tqdm(dp_folders):

        files = glob.glob(f'{dp_folder}/*.png')
        for file in files:
            os.remove(file)

        nb_path = dp_folder / "split_data_cut.ipynb"
        gather_plot_from_nb(nb_path)


def split_noteboook(notebook_path):
    '''
    Split notebook cell to the data reading and convering cell and plotting df dataframe
    '''

    data_string = 'import pandas as pd\n' + 'df = pd.read_csv("data.csv")\n'
    lines_to_remove = {'import pandas as pd\n', 'df = pd.read_csv("data.csv")\n'}

    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    code = nb.cells[0]['source']
    code_new = ''.join(line for line in code.splitlines(True) if line not in lines_to_remove)
    nb.cells[0]['source'] = code_new

    new_cell = nbf.v4.new_code_cell(data_string)
    nb.cells.insert(0, new_cell)

    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)


def split_noteboooks(folder):
    dp_folders = get_dp_folders(folder)

    for dp_folder in tqdm(dp_folders):
        nb_path = dp_folder / "split_data_cut.ipynb"
        split_noteboook(nb_path)


def str_in_notebook(folder, substrings):
    '''
    Search a substring in a notebook
    '''

    dp_folders = get_dp_folders(folder)

    for dp_folder in dp_folders:

        nb_path = dp_folder / "split_data_cut.ipynb"

        with open(nb_path) as f:
            nb = nbf.read(f, as_version=4)

        code = nb.cells[1]['source']
        if any(substring in code for substring in substrings):
            print(dp_folder.name)


def generate_df_description(dp_folder):
    os.chdir(dp_folder)
    nb_path = dp_folder / "split_data_cut.ipynb"

    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    code = nb.cells[0]['source']

    locals_dict = {}
    exec(code, {'np': np, 'Circle': Circle}, locals_dict)

    df = locals_dict['df']

    df_descr = get_pycharm_dataframe_description(df)
    df_descr_file = dp_folder / "data_descr.txt"

    with open(df_descr_file, 'w') as f:
        f.write(df_descr)

    return None


def generate_df_description_all(folder):
    dp_folders = get_dp_folders(folder)
    for dp_folder in tqdm(dp_folders):

        try:
            generate_df_description(dp_folder)
        except NameError:
            print(dp_folder.name)

    return None


if __name__ == "__main__":

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)

    data_folder = Path(config.matplotlib_dataset_path)
    validated_notebooks = Path(config.validated_notebooks)
    dataset_valid_step_1 = Path(config.dataset_valid_step_1)
    cut_notebooks_folder = Path(config.dataset_valid_step_1_notebooks)

    # copy_valid_dp(data_folder, validated_notebooks, dataset_valid_step_1)
    # generate_stand_alone_dps(dataset_valid_step_1)
    # gather_nbs(dataset_valid_step_1, cut_notebooks_folder)
    # clean_dp_nbs(dataset_valid_step_1)
    # gather_plots(dataset_valid_step_1)
    # split_noteboooks(dataset_valid_step_1)
    # generate_df_description_all(target_folder)
