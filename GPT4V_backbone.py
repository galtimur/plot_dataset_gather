from omegaconf import OmegaConf
import base64
import requests
from pathlib import Path
from typing import List
import time
import re


config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
openai_token_file = config.openai_token_file

with open(openai_token_file, "r") as f:
    openai_token = f.read()

class GPT4V:

    def __init__(self, api_key: str, system_prompt: str, wait_time = 20, attempts=10):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model_name = "gpt-4-turbo"
        self.model_url = "https://api.openai.com/v1/chat/completions"
        self.system_prompt = system_prompt
        self.wait_time = wait_time
        self.attempts = attempts

    def encode_image(self, image_path):

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_images(self, image_paths: List[str|Path]):

        encoded_images = []
        for image_path in image_paths:
            encoded_images.append(self.encode_image(image_path))

        return encoded_images

    def ask(self, request: str, image_paths: List[str|Path] = [], image_detail: str = "auto"):

        encoded_images = self.encode_images(image_paths)

        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        content = [{"type": "text", "text": request}]

        for encoded_image in encoded_images:
            content_image = {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}",
                                           "detail": image_detail}}
            content.append(content_image)

        messages.append({"role": "user", "content": content})
        payload = {"model": self.model_name,
                   "messages": messages}

        response = requests.post(self.model_url, headers=self.headers, json=payload)

        return response.json()
    def make_request(self, request, image_paths: List[str|Path] = [], image_detail: str = "auto"):
        error_counts = 0
        while error_counts<self.attempts:
            response = self.ask(request=request, image_paths=image_paths, image_detail=image_detail)

            if "error" not in response.keys():
                break
            else:
                message = response["error"]["message"]
                seconds_to_wait = re.search(r'Please try again in (\d+)s\.', message)
                if seconds_to_wait is not None:
                    wait_time = 1.5*int(seconds_to_wait.group(1))
                    print(f"Waiting {wait_time} s")
                    time.sleep(wait_time)
                else:
                    print(f"Cannot parse retry time from error message. Will wait for {wait_time} seconds")
                    print(message)
                    response = None
                    time.sleep(20)
                    error_counts += 1

        return response