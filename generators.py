from pathlib import Path
from tqdm import tqdm
import json
import re

from data import PlotDataLoader, PlotDataPoint
from benchmark_utils import TaskChanger
from GPT4V_backbone import GPT4V
from LLM_utils import prepare_pipeline

class CodePlotGenerator:

    '''
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
    '''
    def __init__(self, model, output_file: str | Path, plotting_prompt: str = "", system_prompt: str = ""):

        self.model = model
        self.plotting_prompt = plotting_prompt
        self.system_prompt = system_prompt
        self.output_file = Path(output_file)

    def gather_code(self, answer: str) -> str:

        '''
        Gather all python code blocks in a response to a single code
        '''

        pattern = r'```python\n(.*?)\n```'
        code_blocks = re.findall(pattern, answer, re.DOTALL)

        return '\n'.join(code_blocks)

    def generate_plotting_request(self, datapoint: PlotDataPoint, plotting_prompt: str = "") -> str:

        '''
        Request to ask model to write a code for plotting. Add dataframe description
        '''

        task = plotting_prompt
        for i, task_part in enumerate(datapoint.task.values(), start=1):
            task += f"{i}. {task_part.lstrip()}\n"

        task = task.replace("\n\n", "\n")

        return task

    def generate_codeplot(self, datapoint: PlotDataPoint) -> str:

        '''
        Request a model to write a plot code for given datapoint and plotting and system prompt
        Returns raw LLM response text (only message)
        '''

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

    # TODO WIP thats just a copy of previous class

    '''
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
    '''
    def __init__(self, model, output_file: str | Path, plotting_prompt: str = "", system_prompt: str = ""):

        self.model = model
        self.plotting_prompt = plotting_prompt
        self.system_prompt = system_prompt
        self.output_file = Path(output_file)

    def gather_code(self, answer: str) -> str:

        '''
        Gather all python code blocks in a response to a single code
        '''

        pattern = r'```python\n(.*?)\n```'
        code_blocks = re.findall(pattern, answer, re.DOTALL)

        return '\n'.join(code_blocks)

    def generate_plotting_request(self, datapoint: PlotDataPoint, plotting_prompt: str = "") -> str:

        '''
        Request to ask model to write a code for plotting. Add dataframe description
        '''

        task = plotting_prompt
        for i, task_part in enumerate(datapoint.task.values(), start=1):
            task += f"{i}. {task_part.lstrip()}\n"

        task = task.replace("\n\n", "\n")

        return task

    def generate_codeplot(self, datapoint: PlotDataPoint) -> str:

        '''
        Request a model to write a plot code for given datapoint and plotting and system prompt
        Returns raw LLM response text (only message)
        '''

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
