from tqdm import tqdm
import json
from GPT4V_backbone import GPT4V
import re

from LLM_utils import prepare_pipeline, generate_benchmark_request, read_task_responses

def score_by_GPT(results_plot, pipline_parameters):

    gpt4v = GPT4V(api_key=pipline_parameters.openai_token, system_prompt=pipline_parameters.instructs["system prompt"])

    responses = []
    for idx, result in tqdm(results_plot.items()):

        dp_folder = pipline_parameters.dataset_folder / str(idx)
        images = result["images"]

        if len(images)==0:
            print(f"No image for ID {idx}")
            continue

        request, plots = generate_benchmark_request(dp_folder, pipline_parameters.instructs, result)

        response = gpt4v.make_request(request=request, images=plots, image_detail="low")
        response["id"] = idx

        responses.append(response)
        with open(pipline_parameters.output_file, "a") as f:
            json.dump(response, f)
            f.write("\n")

    return responses

def parse_bench_response(message):

    match = re.search(r"[FINAL SCORE]:? ?(\d+)", message)

    if match:
        return int(match.group(1))
    else:
        return None


def gather_scores(responses, results_plot):

    benchmark_results = dict()
    for idx, result in tqdm(results_plot.items()):

        images = result["images"]
        bench = {"score": 0, "error": result["error"]}

        if len(images)==0:
            print(f"No image for ID {idx}")
        else:
            score = parse_bench_response(responses[idx])
            if score is not None:
                bench["score"] = score
            else:
                print(f"Could not parse bench response:\nresponses[idx]")
                bench["score"] = "UNK"

        benchmark_results[idx] = bench

    return benchmark_results


if __name__ == "__main__":

    config_path = "configs/config.yaml"
    out_filename = "benchmark_responses.jsonl"

    pipline_parameters = prepare_pipeline(config_path, out_filename, "prompts/benchmark.json")
    results_file = pipline_parameters.out_folder / "gpt_plots_results.json"
    bench_results_file = pipline_parameters.out_folder / "benchmark_results.json"

    with open(results_file, 'r') as f:
        results_plot = json.load(f)

    # responses = score_by_GPT(results_plot, pipline_parameters)
    responses = read_task_responses(pipline_parameters.output_file)
    benchmark_results = gather_scores(responses, results_plot)

    with open(bench_results_file, 'w') as f:
        json.dump(benchmark_results, f)
