import os
from pathlib import Path
from tqdm import tqdm
import json
import re
from typing import List, Dict
import nbformat as nbf
import subprocess

from data import PlotDataLoader, PlotDataPoint
from utils import read_jsonl


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
        output_file: str | Path,
        dataset_folder: Path | str,
        temp_dir: str | Path = ".temp",
    ) -> None:

        self.temp_dir = Path(temp_dir)
        self.output_file = Path(output_file)
        self.dataset_folder = Path(dataset_folder)
        os.makedirs(self.temp_dir, exist_ok=True)

    def build_plots(self, response_file: str | Path | None = None, responses: List[Dict] | None = None) -> Path:

        """
        Takes either response_file of list of responses.
        List of responses is prioritized

        Gather all datapoints code in a single notebook and run it.
        So, each cell is a datapoint code with output - plot image
        """

        if response_file is None and responses is None:
            raise ValueError("Either response_file or responses must be provided.")

        if response_file is not None and responses is not None:
            print("Both responses file and responses list provided. Responses list would be used.")

        if responses is None:
            responses = read_jsonl(response_file)

        responses_dict = dict()
        for response in responses:
            if "id" in response:
                idx = response["id"]
                code = response["code"]
                responses_dict[idx] = code

        dp_ids = sorted(list(responses_dict.keys()))

        plot_cells = []
        for idx in dp_ids:

            # TODO may be use DataLoader instead
            data_file = self.dataset_folder / str(idx) / "data.csv"
            data_code_file = self.dataset_folder / str(idx) / "data_load.py"

            with open(data_code_file, "r") as f:
                data_load_code = f.read()
            data_load_code = data_load_code.replace("data.csv", str(data_file))

            generated_code = responses_dict[idx]

            # Gather a code, adding index number at the first line as comment
            # and resetting all variables at the last line
            plot_code_nb = "\n".join(
                [f"# id = {idx}", data_load_code, generated_code, "%reset -f"]
            )

            plot_cells.append(plot_code_nb)

        plots_nb_path = self.temp_dir / "all_plots.ipynb"

        build_new_nb(plot_cells, plots_nb_path)
        print("Running all codes to build plots")
        cmd = f'jupyter nbconvert --execute --allow-errors --to notebook --inplace "{plots_nb_path}"'
        subprocess.call(cmd, shell=True)

        return plots_nb_path
