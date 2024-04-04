import os
import shutil
from tqdm import tqdm
import subprocess
import nbformat as nbf
from nbformat.v4 import new_markdown_cell
import json
import base64
from omegaconf import OmegaConf

from utils import get_dp_folders

# %%

def clear_notebook(notebook_path):
    '''
    Remove all non-code cells and first cell, which is only '%matplotlib inline'
    Also, if there is savefig command, delete it and replace it by plt.show()
    '''

    with open(notebook_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    index = 0
    for cell in nb['cells']:

        if cell['cell_type'] == 'code':

            if index == 0 and cell['source'].strip() != '%matplotlib inline':
                new_cells = cell
                break

            if index == 1:
                new_cells = cell
                break

            index += 1

    # replace cells with our modified cells

    cell_cleared = []
    add_show = False
    for line in new_cells['source'].split('\n'):
        if 'plt.savefig' in line:
            add_show = True
        else:
            cell_cleared.append(line)

    if add_show:
        cell_cleared.append('plt.show()')

    new_cells['source'] = '\n'.join(cell_cleared)

    nb['cells'] = [new_cells]

    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)


def copy_and_clean(source_folder_path, target_folder_path):
    '''
    traverse source folder and copy notebooks (except cached) into target folder.
    perform cleaning of the notebook
    '''

    for root, dirs, files in os.walk(source_folder_path):

        dirs[:] = [d for d in dirs if d not in ['.ipynb_checkpoints']]
        for file in files:
            if file.endswith(".ipynb"):

                subfolder_name = os.path.basename(root)
                new_file_name = subfolder_name + "__" + file
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_folder_path, new_file_name)

                shutil.copy2(source_file_path, target_file_path)
                if file.startswith("pgf"):
                    print(file)

                try:
                    clear_notebook(target_file_path)
                except UnicodeEncodeError:
                    print(f"Can not cleat NB {file}")
                    os.remove(target_file_path)


def run_notebooks(dataset_folder):
    '''
    traverse all notebooks and run them to get plot image inside it
    '''

    ipynb_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.ipynb')]

    for file in tqdm(ipynb_files):
        cmd = f'jupyter nbconvert --execute --to notebook --inplace "{file}"'
        subprocess.call(cmd, shell=True)


def check_image_existance(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    has_image_output = False

    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell['outputs']:
                if output['output_type'] == 'display_data':
                    if 'image/png' in output['data']:
                        has_image_output = True
                        break

        if has_image_output:
            break

    return has_image_output


def filter_nb_by_image(dataset_folder):
    ipynb_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.ipynb')]

    for file in ipynb_files:
        is_out_image = check_image_existance(file)
        if not is_out_image:
            os.remove(file)


def seaprate_by_folders(dataset_folder):
    ipynb_files = [f for f in os.listdir(dataset_folder) if f.endswith('.ipynb')]
    ipynb_files_full = [os.path.join(dataset_folder, f) for f in ipynb_files]

    for i, (notebook_name, notebook_path) in enumerate(zip(ipynb_files, ipynb_files_full)):
        new_folder_path = os.path.join(dataset_folder, str(i))
        os.makedirs(new_folder_path, exist_ok=True)
        new_notebook_path = os.path.join(new_folder_path, "plot.ipynb")

        shutil.copy(notebook_path, new_notebook_path)

        plot_class, plot_name = notebook_name[:-6].split("__")

        info = {"plot_class": plot_class, "plot_name": plot_name, "id": i}
        with open(os.path.join(new_folder_path, "info.json"), "w") as json_file:
            json.dump(info, json_file)


def parse_code_and_images(notebook_filepath):
    dp_folder = notebook_filepath.parent

    with open(notebook_filepath, 'r') as f:
        nb = nbf.read(f, as_version=4)

    code_cell = nb.cells[0]

    with open(dp_folder / 'plot.py', 'w') as f:
        f.write(code_cell['source'])

    for output in code_cell['outputs']:
        if output['output_type'] == 'display_data' and 'image/png' in output['data']:
            image_data = output['data']['image/png']
            break

    with open(dp_folder / 'plot.png', 'wb') as f:
        f.write(base64.b64decode(image_data))


def generate_separate_code_and_images(dataset_folder):
    dp_folders = get_dp_folders(dataset_folder)
    print("Extracting code and images from notebooks")

    for folder in tqdm(dp_folders):
        new_notebook_path = folder / "plot.ipynb"
        parse_code_and_images(new_notebook_path)


def add_info_to_nb(dp_folder):
    with open(dp_folder / 'info.json', 'r') as f:
        info_data = json.load(f)

    info_string = json.dumps(info_data)

    md_cell = new_markdown_cell('INFO: ' + info_string)
    md_cell.pop('id', None)

    with open(dp_folder / 'plot.ipynb', 'r') as f:
        nb = nbf.read(f, as_version=4)

    nb.cells.insert(0, md_cell)

    with open(dp_folder / 'plot.ipynb', 'w') as f:
        nbf.write(nb, f)


def add_info_to_nb_datapoints(dataset_folder):
    dp_folders = get_dp_folders(dataset_folder)

    for dp_folder in tqdm(dp_folders):
        add_info_to_nb(dp_folder)


# %%

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    dataset_folder = config.matplotlib_dataset_path
    source_folder = config.matplotlib_source_path

    # copy_and_clean(source_folder, dataset_folder)
    # run_notebooks(dataset_folder)
    # filter_nb_by_image(dataset_folder)
    # seaprate_by_folders(dataset_folder)
    # generate_separate_code_and_images(dataset_folder)
    # add_info_to_nb_datapoints(dataset_folder)

# %%
