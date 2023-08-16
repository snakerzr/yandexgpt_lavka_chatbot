from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import requests
import json
import time
import jwt
import os
from dotenv import find_dotenv, load_dotenv


class YandexGPT(LLM):
    model: str = "general"
    partial_results: bool = True
    temperature: float = 0.01
    max_tokens: int = 7400

    service_account_id = ""
    key_id = ""
    private_key = ""

    token: str = ""
    token_expiration_time: int = 0

    # last_request_time: int = 0

    def __init__(self, load_env_from_file=False):
        super().__init__()

        if load_env_from_file:
            load_dotenv(find_dotenv())

        self.service_account_id = os.environ["SERVICE_ACCOUNT_ID"]
        self.key_id = os.environ["ID"]
        self.private_key = os.environ["PRIVATE_KEY"]

    def _generate_token(self, expiration_time_delta=360):
        now = int(time.time())
        payload = {
            "aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            "iss": self.service_account_id,
            "iat": now,
            "exp": now + expiration_time_delta,
        }

        # JWT generation
        encoded_token = jwt.encode(
            payload, self.private_key, algorithm="PS256", headers={"kid": self.key_id}
        )

        url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
        x = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"jwt": encoded_token},
        ).json()
        token = x["iamToken"]

        self.token = token
        self.token_expiration_time = now + expiration_time_delta

    def _check_token_expiration_time(self):
        now = int(time.time())
        if self.token_expiration_time < now:
            self._generate_token()

    def _request(self, instruction_text="", request_text=""):
        self._check_token_expiration_time()
        time.sleep(1)
        # URL to access the model
        url = "https://api.ml.yandexcloud.net/llm/v1alpha/instruct"

        # Building a prompt
        data = {}
        data["model"] = self.model

        # Specify an instruction for YandexGPT
        data["instruction_text"] = instruction_text

        # Set up advanced model parameters
        data["generation_options"] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Enter the request text
        data["request_text"] = request_text

        # Get the model's response
        response = requests.post(
            url, headers={"Authorization": "Bearer " + self.token}, json=data
        )

        if response.status_code == 200:
            # print(response.json())
            # print(type(response.json()))
            return response.json()["result"]["alternatives"][0]["text"]
        else:
            raise ValueError(
                f"Failed to generate response. Status code: {response.status_code}, Response: {response.text}"
            )

    @property
    def _llm_type(self) -> str:
        return "YandexGPT"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response = self._request(request_text=prompt)
        if stop is not None:
            # raise ValueError("stop kwargs are not permitted.")
            print(stop)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "llm_type": self._llm_type,
            "model": self.model,
            "partial_results": self.partial_results,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


if __name__ == "__main__":
    llm = YandexGPT()
    print(llm("Hello!"))
