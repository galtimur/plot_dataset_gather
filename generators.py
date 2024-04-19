import os
from pathlib import Path
from tqdm import tqdm
import json
import re
from typing import List, Dict
import nbformat as nbf
import subprocess

from data import PlotDataLoader, PlotDataPoint
from utils import read_jsonl, save_jsonl


def build_new_nb(blocks: list, nb_path):

    '''
    save codeblocks into notebook
    '''

    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(block) for block in blocks]

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)

class CodePlotGenerator:

    """
    Object that requests to write a code for the plotting the plot.
    At init pass:

    model: the model that has method:
        make_request(task: str, system_prompt: str) -> response
        task: the task for the model, used as user input
        system_prompt: system prompt
        response: dict. {"response": response text, ... any_other_meta_info}

        NOTE that we than parse response to get a code. We assume that code block is marked as following:
        ```python
        CODE
        ```

    output_file: output file to log model responses
    plotting_prompt: Additional plotting prompt, prepended to the plot task. See example in prompts/plot_gen.json
    system_prompt: system prompt for the model
    """

    def __init__(
        self,
        model,
        output_file: str | Path,
        plotting_prompt: str = "",
        system_prompt: str = "",
    ):
        self.model = model
        self.plotting_prompt = plotting_prompt
        self.system_prompt = system_prompt
        self.output_file = Path(output_file)

    def gather_code(self, answer: str) -> str:
        """
        Gather all python code blocks in a response to a single code
        """

        pattern = r"```python\n(.*?)\n```"
        code_blocks = re.findall(pattern, answer, re.DOTALL)

        return "\n".join(code_blocks)

    def generate_plotting_request(
        self, datapoint: PlotDataPoint, plotting_prompt: str = ""
    ) -> str:
        """
        Request to ask model to write a code for plotting. Add dataframe description
        """

        task = plotting_prompt
        for i, task_part in enumerate(datapoint.task.values(), start=1):
            task += f"{i}. {task_part.lstrip()}\n"

        task = task.replace("\n\n", "\n")

        return task

    def generate_codeplot(self, datapoint: PlotDataPoint) -> str:
        """
        Request a model to write a plot code for given datapoint and plotting and system prompt
        Returns raw LLM response text (only message)
        """

        task = self.generate_plotting_request(datapoint, self.plotting_prompt)
        response = self.model.make_request(task, self.system_prompt)

        if response is None:
            print(f"Skipping dp {datapoint.id}")
            return None

        response["id"] = datapoint.id
        response["task"] = task
        response["code"] = self.gather_code(response["response"])

        return response

    def generate_codeplot_datapoints(self, dataset: PlotDataLoader):

        print("Requesting the model to write a code for plots")
        responses = []
        for item in tqdm(dataset):
            # TODO refactor
            # if item.id in pipline_parameters.existing_ids:
            #     continue

            response = self.generate_codeplot(item)

            if response is None:
                print(f"Skipping dp {item.id}")
                continue

            responses.append(response)

            with open(self.output_file, "a") as f:
                json.dump(response, f)
                f.write("\n")

        return responses


class VisGenerator:
    # TODO WIP

    """
    Object that runs generated code to build a plot.
    # At init pass:

    # output_file: output file to log model responses
    """

    def __init__(
        self,
        dataset: PlotDataLoader,
        output_file: str | Path,
        dataset_folder: Path | str,
        temp_dir: str | Path = ".temp",
    ) -> None:

        self.temp_dir = Path(temp_dir)
        self.output_file = Path(output_file)
        self.dataset_folder = Path(dataset_folder)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.dataset = dataset

    def build_plots(self, responses_file: str | Path | None = None, responses: List[Dict] | None = None) -> Path:

        """
        Takes either response_file of list of responses.
        List of responses is prioritized

        Gather all datapoints code in a single notebook and run it.
        So, each cell is a datapoint code with output - plot image
        """

        if responses_file is None and responses is None:
            raise ValueError("Either response_file or responses must be provided.")

        if responses_file is not None and responses is not None:
            print("Both responses file and responses list provided. Responses list would be used.")

        if responses is None:
            responses = read_jsonl(responses_file)
        if responses_file is not None:
            self.responses_file = responses_file

        self.responses = responses

        responses_dict = dict()
        for response in self.responses:
            if "id" in response:
                idx = response["id"]
                code = response["code"]
                responses_dict[idx] = code

        dp_ids = list(responses_dict.keys())

        plot_cells = []
        for item in self.dataset:

            idx = item.id
            if not idx in dp_ids:
                continue

            data_load_code = item.code_data.replace("data.csv", str(item.dp_path / "data.csv"))
            generated_code = responses_dict[idx]

            # Gather a code, adding index number at the first line as comment
            # and resetting all variables at the last line
            plot_code_nb = "\n".join(
                [f"# id = {idx}", data_load_code, generated_code, "%reset -f"]
            )

            plot_cells.append(plot_code_nb)

        self.plots_nb_path = self.temp_dir / "all_plots.ipynb"

        build_new_nb(plot_cells, self.plots_nb_path)
        print("Running all codes to build plots")
        cmd = f'jupyter nbconvert --execute --allow-errors --to notebook --inplace "{self.plots_nb_path}"'
        subprocess.call(cmd, shell=True)

        return self.plots_nb_path

    def parse_plots_notebook(self, plots_nb_path: Path | None = None) -> Dict:

        '''
        Parses notebook with plotted plots and gathers the results to a json
        '''

        if plots_nb_path is None:
            if hasattr(self, 'plots_nb_path'):
                plots_nb_path = self.plots_nb_path
            else:
                raise ValueError("Either plots_nb_path argument or attribute should exist.")

        with open(plots_nb_path) as f:
            nb = nbf.read(f, as_version=4)

        plot_results = dict()
        for cell_num, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

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
            plot_results[idx] = cell_res

        # Update responses
        for response in self.responses:
            if "id" in response:
                response["plot results"] = plot_results[response["id"]]

        # TODO add safe save
        if os.path.exists(self.output_file):
            print(f"Output file would be overwritten! {self.output_file}")
        save_jsonl(self.responses, self.output_file)
        print(f"Responses and plots are saved in {self.output_file}")

        return self.responses

    def draw_plots(self, responses_file: str | Path | None = None, responses: List[Dict] | None = None) -> Dict:

        self.build_plots(responses_file, responses)
        responses = self.parse_plots_notebook()

        return responses
