import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from utils import read_jsonl


class VisJudge:
    def __init__(
        self,
        vis_judge_model,
        prompts_path: Path | str,
        output_file_judge: Path | str,
        output_file_score: Path | str,
        dataset_folder: Path | str,
    ) -> None:
        self.vis_judge_model = vis_judge_model
        self.dataset_folder = Path(dataset_folder)
        self.output_file_judge = output_file_judge
        self.output_file_score = output_file_score
        with open(prompts_path, "r") as f:
            self.prompts = json.load(f)

    def generate_images_request(self, dp_folder: Path, images: List[str]) -> List:
        "Request to ask model to write a code for plotting. Add dataframe description"

        plot_files = glob.glob(os.path.join(str(dp_folder), "*.png"))
        plot_file_gt = Path(plot_files[0])
        plot_gen = images[0]
        plots = [plot_gen, plot_file_gt]

        return plots

    def score_by_GPT(self, results_plot: dict) -> List[Dict]:
        responses = []
        for result in results_plot:
            if "id" not in result:
                continue
            idx = result["id"]
            images = result["plot results"]["images"]

            if len(images) == 0:
                # TODO make 0 score for that
                print(f"No image for ID {idx}")
                continue

            dp_folder = self.dataset_folder / str(idx)
            plots = self.generate_images_request(dp_folder, images)

            response = self.vis_judge_model.make_request(
                request=self.prompts["request judge"], images=plots, image_detail="auto"
            )

            response["id"] = idx

            responses.append(response)
            with open(self.output_file_judge, "a") as f:
                json.dump(response, f)
                f.write("\n")

        return responses

    def parse_bench_response(self, message):
        match = re.search(r"[FINAL SCORE]:? ?(\d+)", message)

        if match:
            return int(match.group(1))
        else:
            return None

    def calc_bench_stat(self, benchmark_results: dict) -> dict:
        scores = np.array([entry["score"] for entry in benchmark_results.values()])
        errors = np.array([entry["error"] for entry in benchmark_results.values()])

        err_num = sum(1 for error in errors if len(error) > 0)

        statistics = {
            "min score": min(scores),
            "max score": max(scores),
            "mean score": np.mean(scores),
            "median score": np.median(scores),
            "num items": len(benchmark_results),
            "error number": err_num,
            "error rate": err_num / len(benchmark_results),
        }

        return statistics

    def get_benchmark_scores(
        self,
        results_plot: dict,
        scoring_responses: List[Dict] | None = None,
        scoring_responses_file: str | Path | None = None,
    ) -> Tuple[dict, dict]:
        if scoring_responses is None and scoring_responses_file is None:
            scoring_responses = self.score_by_GPT(results_plot)
        elif scoring_responses is None:
            scoring_responses = read_jsonl(scoring_responses_file)
        elif scoring_responses is not None and scoring_responses_file is not None:
            print("You passed both function path and scoring responses list")
            print("The list would be used")

        scoring_responses_dict = {
            response["id"]: response for response in scoring_responses
        }

        benchmark_results = dict()
        for responses in results_plot:
            if "id" not in responses:
                continue
            idx = responses["id"]

            bench = {"score": 0, "error": responses["plot results"]["error"]}

            if len(responses["plot results"]["images"]) == 0:
                print(f"No image for ID {idx}")
            else:
                score = self.parse_bench_response(
                    scoring_responses_dict[idx]["response"]
                )
                if score is not None:
                    bench["score"] = score
                else:
                    print(f"Could not parse bench response:\nresponses[idx]")
                    bench["score"] = "UNK"

            benchmark_results[idx] = bench

        with open(self.output_file_score, "w") as f:
            json.dump(benchmark_results, f)

        bench_stat = self.calc_bench_stat(benchmark_results)

        return benchmark_results, bench_stat
