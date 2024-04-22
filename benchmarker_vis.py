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

    """
    Class for running visual benchmark over the plotted plots comparing with golden truth datapoints
    Visual benchmarking is asking model to compare two images and return a score
    """

    def __init__(
        self,
        vis_judge_model,
        prompts_path: Path | str,
        output_file_bench: Path | str,
        bench_stat_file: Path | str,
        dataset_folder: Path | str,
    ) -> None:
        self.vis_judge_model = vis_judge_model
        self.dataset_folder = Path(dataset_folder)
        self.output_file_bench = output_file_bench
        self.bench_stat_file = bench_stat_file
        with open(prompts_path, "r") as f:
            self.prompts = json.load(f)

    def generate_images_request(self, dp_folder: Path, images: List[str]) -> List:
        # Build a list of plots to compare

        plot_files = glob.glob(os.path.join(str(dp_folder), "*.png"))
        plot_file_gt = Path(plot_files[0])
        plot_gen = images[0]
        plots = [plot_gen, plot_file_gt]

        return plots

    def parse_bench_response(self, message):
        match = re.search(r"[FINAL SCORE]:? ?(\d+)", message)

        if match:
            return int(match.group(1))
        else:
            return None

    def score_by_LLM(self, results_plot: dict) -> List[Dict]:
        """
        Score each plotted plot by LLM comparing with baseline image
        """

        print("Running plots scoring")
        print(f"Results would be saved in {self.output_file_bench }")
        with open(self.output_file_bench, "a") as f:
            json.dump({"request judge":self.prompts["request judge"]}, f)
            f.write("\n")

        benchmark_results = []
        for result in tqdm(results_plot):
            if "id" not in result:
                continue
            idx = result["id"]
            images = result["plot results"]["images"]

            bench = {
                "id": idx,
                "score": 0,
                "has plot": True,
                "error": result["plot results"]["error"],
                "raw response": None,
            }
            if len(images) == 0:
                print(f"No image for ID {idx}")
                bench["has plot"] = False
                benchmark_results.append(bench)
                continue

            dp_folder = self.dataset_folder / str(idx)
            plots = self.generate_images_request(dp_folder, images)

            response = self.vis_judge_model.make_request(
                request=self.prompts["request judge"], images=plots, image_detail="auto"
            )

            score = self.parse_bench_response(response["response"])

            bench["raw response"] = response
            bench["score"] = score
            benchmark_results.append(bench)
            with open(self.output_file_bench, "a") as f:
                json.dump(bench, f)
                f.write("\n")

        return benchmark_results

    def calc_bench_stat(self, benchmark_results: List[dict]) -> dict:
        """
        Calculate statistics of the scores
        """

        benchmark_results = [entry for entry in benchmark_results if "id" in entry]
        scores = np.array(
            [
                entry["score"]
                for entry in benchmark_results
                if isinstance(entry["score"], int)
            ]
        )
        errors = np.array(
            [
                entry["error"]
                for entry in benchmark_results
                if isinstance(entry["score"], int)
            ]
        )

        err_num = sum(1 for error in errors if len(error) > 0)
        num_unparsed = len(benchmark_results) - len(scores)

        statistics = {
            "min score": int(min(scores)),
            "max score": int(max(scores)),
            "mean score": np.mean(scores),
            "median score": np.median(scores),
            "num items": len(benchmark_results),
            "error number": err_num,
            "error rate": err_num / len(benchmark_results),
            "unparsed": num_unparsed,
        }

        return statistics

    def get_benchmark_scores(
        self,
        results_plot: dict | None = None,
        benchmark_results: List[Dict] | None = None,
        benchmark_results_file: str | Path | None = None,
    ) -> Union[Tuple[List, dict], Tuple[None, None]]:
        if benchmark_results is None and benchmark_results_file is None:
            if results_plot is None:
                print("Nothing to analyze is provided")
                return None, None
            benchmark_results = self.score_by_LLM(results_plot)
        elif benchmark_results is None:
            benchmark_results = read_jsonl(benchmark_results_file)
        elif benchmark_results is not None and benchmark_results_file is not None:
            print("You passed both function path and scoring responses list")
            print("The list would be used")

        bench_stat = self.calc_bench_stat(benchmark_results)

        with open(self.bench_stat_file, "w") as f:
            json.dump(bench_stat, f)

        return benchmark_results, bench_stat
