import json
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from data import get_dp_folders
from openai_backbone import ChatMessage, OpenAIBackbone

config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
openai_token_file = config.openai_token_file
dataset_folder = config.matplotlib_dataset_path

output_file = Path(dataset_folder) / "gpt_response.jsonl"

with open(openai_token_file, "r") as f:
    openai_token = f.read()

chat = OpenAIBackbone(model_name="gpt-4", api_key=openai_token)
system_prompt = "You are a helpful programming assistant proficient in python, matplotlib and pandas dataframes. All used variables should be defined. In the end you code should run without exceptions."
instruction = """Change code, so that all data before plotting is gathered to a single dataframe named "df".
  Your response MUST contain EXACTLY TWO codeblocks.
  First for the dataframe construction.
  Second separate codeblock for the plotting the dataframe."""
chat_prompt = ChatMessage(role="system", content=system_prompt)
prompts = {"system prompt": system_prompt, "instruction": instruction}
with open(output_file, "a") as f:
    json.dump(prompts, f)
    f.write("\n")

dp_folders = get_dp_folders(dataset_folder)
responses = []
for i, dp_folder in tqdm(enumerate(dp_folders), total=len(dp_folders)):
    index = int(dp_folder.name)
    code_file = dp_folder / "plot.py"
    with open(code_file, "r") as f:
        code = f.read()

    message = f"Here is a plotting code. {code} \n{instruction}"

    chat_message = ChatMessage(role="user", content=message)

    request = [chat_prompt, chat_message]

    response = chat.generate_msg(request)
    response["id"] = index
    responses.append(response)

    with open(output_file, "a") as f:
        json.dump(response, f)
        f.write("\n")

    # if i > 3-1:
    #     break
