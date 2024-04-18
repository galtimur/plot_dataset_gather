import json

from data import PlotDataLoader
from benchmark_utils import TaskChanger
from GPT4V_backbone import GPT4V
from LLM_utils import prepare_pipeline
from generators import CodePlotGenerator
from user_api import get_pycharm_dataframe_description


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    out_filename = "gpt_plots_test.jsonl"
    pipline_parameters = prepare_pipeline(
        config_path, out_filename, "prompts/plot_gen.json"
    )

    with open(pipline_parameters.output_file, "a") as f:
        json.dump(pipline_parameters.instructs, f)
        f.write("\n")

    gpt4v = GPT4V(
        api_key=pipline_parameters.openai_token,
        system_prompt=pipline_parameters.instructs["system prompt"],
    )

    task_changer = TaskChanger(data_descr_changer = get_pycharm_dataframe_description)
    dataset = PlotDataLoader(
        pipline_parameters.dataset_folder, shuffle=False, task_changer=task_changer
    )

    code_generator = CodePlotGenerator(
        model=gpt4v,
        output_file=pipline_parameters.output_file,
        plotting_prompt=pipline_parameters.instructs["plot instruct"],
        system_prompt=pipline_parameters.instructs["system prompt"],
    )

    dataset = dataset[0:2]
    responses = code_generator.generate_codeplot_datapoints(dataset)
    print(1)
