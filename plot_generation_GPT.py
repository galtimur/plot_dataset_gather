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
    # TODO: all other paths should be in config, lets avoid hardcoded stuff. Also
    out_filename = "gpt_plots_dev.jsonl"
    bench_results_filename = "benchmark_results.jsonl"
    bench_stat_filename = "benchmark_stat.json"
    plot_gen_prompt_file = "prompts/plot_gen.json"

    # TODO: I would have discussion here - current prepare_pipeline feels clunky, maybe we can do it better
    pipline_parameters = prepare_pipeline(
        config_path, out_filename, plot_gen_prompt_file
    )

    # TODO: all of this paths could be available directly in the config --
    #  im not sure that we need to construct them explicitly
    bench_results_file = pipline_parameters.out_folder / bench_results_filename
    bench_stat_file = pipline_parameters.out_folder / bench_stat_filename

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
    # TODO: I dont understand this partial. Here i have more general concern -- maybe pycharm_like_data_prompt should
    #  be a part of TaskChanger not independent parameter.
    pycharm_like_data_prompt = partial(
        pycharm_like_data_prompt, prompt=pipline_parameters.instructs["data instruct"]
    )
    # TODO: lets decouple Task Changer and PlotDataLoader.
    #  We can init raw data with RawPlotDataLoader and only then apply TaskChanger
    #  to change tasks and get final dataset as result
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

        # For dev purposes
        ids_to_test = [19, 20, 45, 62, 77, 96, 97, 107, 108, 109, 135, 137, 142, 144, 154, 186, 195, 211, 260, 299]
        # TODO: Could be oneliner
        dataset_filtered = []
        for item in dataset:
            if item.id in ids_to_test:
                dataset_filtered.append(item)

        # dataset = dataset[0:2]  # For dev purposes
        dataset = dataset_filtered

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
        # plot_responses = read_jsonl(pipline_parameters.output_file)  # For dev purposes
        judge = VisJudge(
            vis_judge_model=gpt4v,
            prompts_path="prompts/benchmark.json",
            output_file_bench=bench_results_file,
            bench_stat_file=bench_stat_file,
            dataset_folder=pipline_parameters.dataset_folder,
        )
        benchmark_results, bench_stat = judge.get_benchmark_scores(
            results_plot_file=pipline_parameters.output_file
        )  # results_plot=plot_responses
        # benchmark_results, bench_stat = judge.get_benchmark_scores(
        #     benchmark_results_file=bench_results_file
        # )

        print(bench_stat)
