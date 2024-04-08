import os
from pathlib import Path
from omegaconf import OmegaConf
import shutil
from tqdm import tqdm
from utils import get_dp_folders

if __name__ == "__main__":

    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    dataset_folder = Path(config.matplotlib_dataset_path)

    dp_folders = get_dp_folders(dataset_folder)

    base_folder = dataset_folder.parent
    valid_dp_folder = base_folder / "valid_dp"
    invalid_dp_folder = base_folder / "invalid_dp"

    os.makedirs(valid_dp_folder, exist_ok=True)
    os.makedirs(invalid_dp_folder, exist_ok=True)

    required_files = ["data_block.py", "data.csv", "info.json", "plot.ipynb", "plot.png", "plot.py", "split_data.ipynb"]

    for dp_folder in tqdm(dp_folders):

        idx = int(dp_folder.name)
        existing_files = os.listdir(dp_folder)
        all_files_present = all(file in existing_files for file in required_files)

        if all_files_present:
            shutil.copy(dp_folder / "split_data.ipynb", valid_dp_folder / f"split_data_{idx}.ipynb")
        else:
            shutil.copy(dp_folder / "split_data.ipynb", invalid_dp_folder / f"split_data_{idx}.ipynb")




