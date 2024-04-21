import json
from functools import partial

from benchmark_utils import TaskChanger
from benchmarker_vis import VisJudge
from data import PlotDataLoader
from generators import CodePlotGenerator, VisGenerator
from GPT4V_backbone import GPT4V
from LLM_utils import prepare_pipeline
from user_api import pycharm_like_data_prompt
from utils import read_jsonl

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    out_filename = "gpt_plots_test.jsonl"
    pipline_parameters = prepare_pipeline(
        config_path, out_filename, "prompts/plot_gen.json"
    )

    generate_code = False
    draw_plots = False
    run_benchmark = True

    # 0. Initialize the model
    # add_args = {"temperature": 0.3}
    add_args = dict()
    gpt4v = GPT4V(
        api_key=pipline_parameters.openai_token,
        system_prompt=pipline_parameters.instructs["system prompt"],
        add_args=add_args,
    )

    # 1. Get dataset
    pycharm_like_data_prompt = partial(
        pycharm_like_data_prompt, prompt=pipline_parameters.instructs["data instruct"]
    )
    task_changer = TaskChanger(data_descr_changer=pycharm_like_data_prompt)
    dataset = PlotDataLoader(
        pipline_parameters.dataset_folder, shuffle=False, task_changer=task_changer
    )

    # 2. Run code generation task
    if generate_code:
        code_generator = CodePlotGenerator(
            model=gpt4v,
            output_file=pipline_parameters.output_file,
            plotting_prompt=pipline_parameters.instructs["plot instruct"],
            system_prompt=pipline_parameters.instructs["system prompt"],
        )

        dataset = dataset[0:2]  # For dev purposes
        with open(pipline_parameters.output_file, "a") as f:
            json.dump(pipline_parameters.instructs, f)
            f.write("\n")
        responses = code_generator.generate_codeplot_datapoints(dataset)

    # 3. Draw plots and save them.
    if draw_plots:
        responses = read_jsonl(pipline_parameters.output_file)  # For dev purposes
        plot_generator = VisGenerator(
            dataset=dataset,
            output_file=pipline_parameters.output_file,
            temp_dir=".temp",
        )

        responses = plot_generator.draw_plots(
            responses_file=pipline_parameters.output_file, responses=None
        )

    # 4. Run benchmarking.
    if run_benchmark:
        plot_responses = read_jsonl(pipline_parameters.output_file)  # For dev purposes
        judge_file = pipline_parameters.output_file.parent / "gpt_response_judge.jsonl"
        output_file_score = (
            pipline_parameters.output_file.parent / "benchmark_scores.json"
        )
        judge = VisJudge(
            vis_judge_model=gpt4v,
            prompts_path="prompts/benchmark.json",
            output_file_judge=judge_file,
            output_file_score=output_file_score,
            dataset_folder=pipline_parameters.dataset_folder,
        )
        scores, stat = judge.get_benchmark_scores(
            results_plot=plot_responses, scoring_responses_file=judge_file
        )

        pass
        print(1)
