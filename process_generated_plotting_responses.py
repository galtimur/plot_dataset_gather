from pathlib import Path
import nbformat as nbf
import json
import re
from omegaconf import OmegaConf
import subprocess

from LLM_utils import read_task_responses
from notebook_utils import build_new_nb

'''

'''

def build_plots(response_path: Path, out_folder: Path, dataset_folder: Path):

    '''
    For each datapoint response runs all the code from it and build plots from it.
    This is done in a single notebook.
    '''

    response = read_task_responses(response_path)
    dp_ids = sorted(list(response.keys()))

    plot_cells = []
    generated_code_dict = {}
    for idx in dp_ids:

        data_file = dataset_folder / str(idx) / "data.csv"
        data_code_file = dataset_folder / str(idx) / "data_load.py"

        with open(data_code_file, 'r') as f:
            data_load_code = f.read()
        data_load_code = data_load_code.replace("data.csv", str(data_file))

        generated_code = response[idx]["code"] # TODO can be an error. Test
        plot_code_nb = "\n".join([f"# id = {idx}", data_load_code, generated_code, "%reset -f"])
        generated_code_dict[idx] = generated_code

        plot_cells.append(plot_code_nb)

    plots_nb_path = out_folder/ "all_plots.ipynb"
    build_new_nb(plot_cells, plots_nb_path)
    cmd = f'jupyter nbconvert --execute --allow-errors --to notebook --inplace "{plots_nb_path}"'
    subprocess.call(cmd, shell=True)

    return plots_nb_path, generated_code_dict

def parse_plots_notebook(notebook_path: Path, generated_code_dict: dict):

    '''
    Parses notebook with plotted plots and gathers the results to a json
    '''

    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    results = dict()

    for cell_num, cell in enumerate(nb.cells):
        if cell.cell_type == "code":

            images = []
            img_num = 0
            # At the beginning of each cell I added "id = {index}".
            # Here we extract this index
            code = cell['source'].lstrip("\n")
            idx = int(code.split("\n")[0].lstrip("# id = "))
            cell_res = {"error": ""}

            for output in cell["outputs"]:
                if output.output_type == "error":
                    cell_res["error"] = output.ename + ": " + output.evalue
                elif output.output_type == "display_data" and "image/png" in output.data:
                    image = output.data["image/png"]
                    images.append(image)
                    img_num += 1

            cell_res["images"] = images
            cell_res["code"] = generated_code_dict[idx]
            results[idx] = cell_res

    return results

def parse_llm_plot_results(config_path):

    config = OmegaConf.load(config_path)
    dataset_folder = Path(config.dataset_final)
    out_folder = Path(config.out_folder)
    response_path = out_folder / "gpt_plots_responses.jsonl"
    results_path = out_folder / "gpt_plots_results.json"

    print("Building plots ...")
    plots_nb_path, generated_code_dict = build_plots(response_path, out_folder, dataset_folder)
    # plots_nb_path = out_folder / "all_plots.ipynb"
    results = parse_plots_notebook(plots_nb_path, generated_code_dict)

    with open(results_path, "w") as f:
        json.dump(results, f)

    print(f"Parsed results saved in {str(results_path)}...")

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    parse_llm_plot_results(config_path)

pass
