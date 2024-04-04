from openai_backbone import OpenAIBackbone, ChatMessage
from omegaconf import OmegaConf
from pathlib import Path
import tqdm

from utils import get_dp_folders, save_jsonl

config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
openai_token_file = config.openai_token_file
dataset_folder = config.matplotlib_dataset_path

with open(openai_token_file, "r") as f:
    openai_token = f.read()

chat = OpenAIBackbone(model_name="gpt-4", api_key=openai_token)
chat_prompt = ChatMessage(role="system", content="You are a helpful programming assistant proficient in python, matplotlib and pandas dataframes.")

dp_folders = get_dp_folders(dataset_folder)
responses = []
for i, dp_folder in tqdm(enumerate(dp_folders)):

    index = int(dp_folder.name)
    code_file = dp_folder / "plot.py"
    with open(code_file, "r") as f:
        code = f.read()

    message = f"""Here is a plotting code. {code} \n Change code, so all the input to be a dataframe.
     Name dataframes data_i, where i is order of appearance in the code.
      In your response make codeblock for the data separate of the plotting codeblock.
      Separate them by words PLOTTING BLOCK """

    chat_message = ChatMessage(role="user", content=message)

    request = [chat_prompt, chat_message]

    response = chat.generate_msg(request)
    # print(response)
    response["id"] = index
    responses.append(response)

    if i > 15-1:
        break

save_jsonl(responses, Path(dataset_folder) / "gpt_response.jsonl")
