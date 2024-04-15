from typing import List

from tqdm import tqdm
import json
from GPT4V_backbone import GPT4V
import re
import random
from pathlib import Path
from scipy.special import softmax
import numpy as np

from utils import get_dp_folders, read_jsonl
from LLM_utils import prepare_pipeline, generate_benchmark_request, read_task_responses


def get_random_dp_folder(dataset_folder, target_idx):
    dp_ids = [folder.name for folder in get_dp_folders(dataset_folder)]
    random.shuffle(dp_ids)

    for idx in dp_ids:
        if target_idx != idx:
            return Path(dataset_folder, idx)


def score_by_GPT(
    results_plot, pipline_parameters, do_random: bool = False, do_logprobs: bool = False
):
    add_args = {"max_tokens": 1, "temperature": 0.3}
    tokens_highlighted = [str(i) for i in range(11)]
    gpt4v = GPT4V(
        api_key=pipline_parameters.openai_token,
        system_prompt=pipline_parameters.instructs["system prompt"],
        do_logprobs=do_logprobs,
        add_args=add_args,
        tokens_highlighted=tokens_highlighted,
    )

    responses = []
    for idx, result in tqdm(results_plot.items()):
        if not do_random:
            dp_folder = pipline_parameters.dataset_folder / str(idx)
        else:
            dp_folder = get_random_dp_folder(
                pipline_parameters.dataset_folder, str(idx)
            )
            idx_random = dp_folder.name

        images = result["images"]

        if len(images) == 0:
            print(f"No image for ID {idx}")
            continue

        request, plots = generate_benchmark_request(
            dp_folder, pipline_parameters.instructs, result
        )

        response = gpt4v.make_request(request=request, images=plots, image_detail="low")
        response["id"] = idx
        response["probs"] = do_logprobs
        if do_random:
            response["id_rnd"] = idx_random

        responses.append(response)
        with open(pipline_parameters.output_file, "a") as f:
            json.dump(response, f)
            f.write("\n")

    return responses, gpt4v.tokens_highlighted


def parse_bench_response(message, tokens_highlighted):
    if isinstance(message, str):
        return parse_bench_response_text(message)
    elif isinstance(message, dict) and "logprobs" in message:
        return parse_bench_response_logprobs(message, tokens_highlighted)


def parse_bench_response_text(message):
    match = re.search(r"[FINAL SCORE]:? ?(\d+)", message)

    if match:
        return int(match.group(1))
    else:
        return None

def calc_mean_score(logprobs: List[float], scores: List[int]):

    probs = softmax(logprobs)
    score = np.dot(probs, np.array(scores))

    return score



def parse_bench_response_logprobs(logprobs, tokens_highlighted):
    logprobs = logprobs["logprobs"]
    probs = []
    scores = []
    for entry in logprobs:
        if entry['token'] in tokens_highlighted:
            probs.append(entry['logprob'])
            scores.append(int(entry['token']))
    score = calc_mean_score(probs, scores)

    return score


def gather_scores(responses, results_plot, tokens_highlighted):
    benchmark_results = dict()
    for idx, result in tqdm(results_plot.items()):
        images = result["images"]
        bench = {"score": 0, "error": result["error"]}

        if len(images) == 0:
            print(f"No image for ID {idx}")
        else:
            score = parse_bench_response(responses[idx], tokens_highlighted)
            if score is not None:
                bench["score"] = score
            else:
                print(f"Could not parse bench response:\nresponses[idx]")
                bench["score"] = "UNK"

        benchmark_results[idx] = bench

    return benchmark_results

def ammend_rnd_idx(benchmark_results, benchmark_response_file):
    benchmark_response = read_jsonl(benchmark_response_file)
    for response in benchmark_response:
        idx = response.get("id")
        if idx is not None:
            benchmark_results[idx]["id_rnd"] = response["id_rnd"]
    return benchmark_results


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    do_random = False
    do_logbrobs = True
    results_filename = f"benchmark_results"
    out_filename = f"benchmark_responses"  # {prefix}.jsonl"
    prompt_file_path = "prompts/benchmark.json"
    if do_random:
        results_filename += "_random"
        out_filename += "_random"
    if do_logbrobs:
        results_filename += "_probs"
        out_filename += "_probs"
        prompt_file_path = "prompts/benchmark_probs.json"


    pipline_parameters = prepare_pipeline(
        config_path,
        out_filename=f"{out_filename}.jsonl",
        prompt_file_path=prompt_file_path,
    )
    results_file = pipline_parameters.out_folder / "gpt_plots_results.json"
    bench_results_file = pipline_parameters.out_folder / f"{results_filename}.json"

    with open(results_file, "r") as f:
        results_plot = json.load(f)

    # responses, tokens_highlighted = score_by_GPT(results_plot, pipline_parameters, do_random, do_logbrobs)
    tokens_highlighted = [str(i) for i in range(11)]
    responses = read_task_responses(pipline_parameters.output_file)
    benchmark_results = gather_scores(responses, results_plot, tokens_highlighted)

    if do_random:
        benchmark_results = ammend_rnd_idx(
            benchmark_results, pipline_parameters.output_file
        )

    with open(bench_results_file, "w") as f:
        json.dump(benchmark_results, f)
