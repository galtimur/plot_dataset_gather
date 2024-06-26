import base64
import re
import time
from pathlib import Path
from typing import Dict, List, Union

import requests
import tiktoken
from omegaconf import OmegaConf

config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
openai_token_file = config.openai_token_file

with open(openai_token_file, "r") as f:
    openai_token = f.read()


class GPT4V:
    def __init__(
            self,
            api_key: str,
            system_prompt: str,
            do_logprobs: bool = False,
            tokens_highlighted: List[str] = [],
            add_args: dict = {},
            wait_time=20,
            attempts=10,
    ) -> None:

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model_name = "gpt-4-turbo"
        self.model_url = "https://api.openai.com/v1/chat/completions"
        self.system_prompt = system_prompt
        self.wait_time = wait_time
        self.attempts = attempts
        if do_logprobs:
            self.construct_logit_args(tokens_highlighted)
        else:
            self.args = {}

        # TODO: I have concern about the model arguments - maybe it should be in config?
        #  What if we want to experiment with temperature?
        self.args.update(add_args)
        self.tokens_highlighted = tokens_highlighted

    def construct_logit_args(
            self, tokens_highlighted: List[str] = [], logit_bias_value: float = 30.0
    ) -> None:

        tokenizer = tiktoken.encoding_for_model(self.model_name)

        options_tok_ids = dict()
        logit_bias = dict()

        for opt in tokens_highlighted:
            tok_ids = tokenizer.encode(opt)
            # TODO: lets instead assert we will have if + warning raise with informative message
            assert len(tok_ids) == 1
            logit_bias[tok_ids[0]] = logit_bias_value
            options_tok_ids[opt] = tok_ids


        self.args = {
            "max_tokens": 1,
            "temperature": 0.3,
            "n": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "logit_bias": logit_bias,
        }

    # TODO: static?
    def encode_image(self, image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_images(self, images: List[str | Path]) -> List[str]:
        # Important!
        # If you pass not Path object, but string, it will be read as encoded image
        encoded_images = []
        for image in images:
            if isinstance(image, Path):
                image_encoded = self.encode_image(image)
            else:
                if "/" in image[:10]:
                    # TODO: Maybe there it will be simpler to check if
                    #  path exists and if it is not consider string as image?
                    print(
                        f"Note, you passed string object for the image. It would be considered as encoded image, not path!\nFirst 10 symbols of the image: {image[:10]}"
                    )
                image_encoded = image

            encoded_images.append(image_encoded)

        return encoded_images

    def ask(
            self,
            request: str,
            system_prompt: str | None = None,
            images: List[str | Path] = [],  # TODO in not sure that it should have default value
            image_detail: str = "auto",
    ) -> dict:

        # TODO: by design should it be overwriteble on the fly? Or maybe it is better always use self.system_prompt
        if system_prompt is not None:
            self.system_prompt = system_prompt

        encoded_images = self.encode_images(images)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]
        content = [{"type": "text", "text": request}]
        # TODO: avoid empty operations
        for encoded_image in encoded_images:
            content_image = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": image_detail,
                },
            }
            content.append(content_image)

        messages.append({"role": "user", "content": content})
        payload = {"model": self.model_name, "messages": messages}

        payload.update(self.args)

        response = requests.post(self.model_url, headers=self.headers, json=payload)

        return response.json()

    def make_request(
            self,
            request: str,
            system_prompt: str | None = None,
            images: List[str | Path] = [],
            image_detail: str = "auto",
    ) -> Union[Dict, None]:

        error_counts = 0
        while error_counts < self.attempts:
            response = self.ask(
                request=request,
                system_prompt=system_prompt,
                images=images,
                image_detail=image_detail,
            )

            if "error" not in response.keys():
                response["response"] = response["choices"][0]["message"]["content"]
                response["choices"][0]["message"]["content"] = "MOVED to response key"
                break
            else:
                message = response["error"]["message"]
                seconds_to_wait = re.search(r"Please try again in (\d+)s\.", message)
                if seconds_to_wait is not None:
                    wait_time = 1.5 * int(seconds_to_wait.group(1))
                    print(f"Waiting {wait_time} s")
                    time.sleep(wait_time)
                else:
                    # TODO: Here will be bug because wait_time will not be defined.
                    #  Please rewrite so we get waite time first and do wait after.
                    #  Also error counting happening only in one branch of if-else
                    print(
                        f"Cannot parse retry time from error message. Will wait for {wait_time} seconds"
                    )
                    print(message)
                    response = None
                    time.sleep(20)
                    error_counts += 1

        # TODO: if we have all errors there will be no error and here wil be a bug.
        return response
