import base64
import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

from natsort import natsorted


def get_dp_folders(folder_path: Path) -> List[Path]:
    subfolder_list = []
    dp_folders = natsorted(os.listdir(folder_path))

    for name in dp_folders:
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            subfolder_list.append(Path(full_path))

    return subfolder_list


@dataclass
class PlotDataPoint:
    code_plot: str
    code_data: str
    task: dict
    image: str
    id: int
    dp_path: Path  # TODO is it ok, or better make it str?

# TODO: I would like to have a discussion there. Im not sure that we need to create so low level dataset here.
#  Why we can just use DataFrame?
class PlotRawDataLoader:

    """
    Dataloader for the plot dataset.
    Reads all items from the datapoint folder (code, task, plot image)
    """

    def __init__(self, data_dir: str | Path, shuffle: bool = False) -> None:
        data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.data_points = get_dp_folders(data_dir)
        if shuffle:
            random.shuffle(self.data_points)
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def read_datapoint(self, dp_folder: Path) -> PlotDataPoint:
        idx = int(dp_folder.name)
        code_plot_file = dp_folder / "plot.py"
        code_data_file = dp_folder / "data_load.py"
        task_file = dp_folder / "task.json"
        plot_files = glob.glob(os.path.join(str(dp_folder), "*.png"))

        files_exist = all(
            [
                os.path.exists(file)
                for file in [code_plot_file, code_data_file, task_file]
            ]
        )

        if not files_exist or len(plot_files) == 0:
            raise FileNotFoundError(
                f"Code file not found for data point {str(dp_folder)}"
            )

        plot_file = Path(plot_files[0])
        # TODO: cycle?
        with open(task_file, "r") as f:
            task_dict = json.load(f)
        with open(code_plot_file, "r") as f:
            code_plot = f.read()
        with open(code_data_file, "r") as f:
            code_data = f.read()
        with open(plot_file, "rb") as f:
            image = base64.b64encode(f.read()).decode("utf-8")

        return PlotDataPoint(
            code_plot=code_plot,
            code_data=code_data,
            task=task_dict,
            image=image,
            dp_path=dp_folder,
            id=idx,
        )

    def __next__(self) -> PlotDataPoint:
        if self.current_idx >= len(self.data_points):
            raise StopIteration

        dp_folder = self.data_points[self.current_idx]
        self.current_idx += 1

        return self.read_datapoint(dp_folder)

    def __getitem__(self, index: int) -> PlotDataPoint:
        if isinstance(index, int):
            dp_folder = self.data_points[index]
            return self.read_datapoint(dp_folder)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.data_points))
            sliced_data_points = [
                self.read_datapoint(self.data_points[i])
                for i in range(start, stop, step)
            ]
            return sliced_data_points

    def __len__(self):
        return len(self.data_points)


class PlotDataLoader(PlotRawDataLoader):
    def __init__(
        self,
        data_dir: str | Path,
        shuffle: bool = False,
        # TODO how to annotate task_changer: TaskChanger without circular import?
        task_changer=None,
    ) -> None:
        super().__init__(data_dir, shuffle)
        self.task_changer = task_changer

    def read_datapoint(self, dp_folder: Path) -> PlotDataPoint:
        datapoint = super().read_datapoint(dp_folder)

        if self.task_changer is not None:
            datapoint.task = self.task_changer.change_task(datapoint)

        return datapoint
