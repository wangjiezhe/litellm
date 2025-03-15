"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

from typing import List, Optional, Tuple

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelInfoBase

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LiteLLMProxyChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")  # type: ignore
        dynamic_api_key = api_key or get_secret_str("LITELLM_PROXY_API_KEY")
        return api_base, dynamic_api_key

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base, api_key = self._get_openai_compatible_provider_info(api_base, api_key)
        if api_base is None:
            raise ValueError(
                "api_base not set for LiteLLM Proxy route. Set in env via `LITELLM_PROXY_API_BASE`"
            )
        models = super().get_models(api_key=api_key, api_base=api_base)
        return [f"litellm_proxy/{model}" for model in models]

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("LITELLM_PROXY_API_KEY")

    def get_model_info(
        self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> ModelInfoBase:
        """
        curl http://localhost:4000/model/info
        """
        if model.startswith("litellm_proxy/"):
            model = model.split("/", 1)[1]
        api_base, api_key = self._get_openai_compatible_provider_info(api_base, api_key)
        if api_base is None:
            raise ValueError(
                "api_base not set for LiteLLM Proxy route. Set in env via `LITELLM_PROXY_API_BASE`"
            )

        response = litellm.module_level_client.get(
            url=f"{api_base}/model/info",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get model info: {response.text}")

        for model_info in response.json()["data"]:
            if (
                model_info["model_name"] == model
                or model_info["litellm_params"]["model"] == model
            ):
                _max_tokens: Optional[int] = model_info["model_info"].get(
                    "max_tokens", None
                )
                return ModelInfoBase(
                    key=model,
                    litellm_provider="litellm_proxy",
                    mode="chat",
                    input_cost_per_token=model_info["model_info"].get(
                        "input_cost_per_token", 0.0
                    ),
                    output_cost_per_token=model_info["model_info"].get(
                        "output_cost_per_token", 0.0
                    ),
                    max_tokens=_max_tokens,
                    max_input_tokens=_max_tokens,
                    max_output_tokens=_max_tokens,
                )

        raise Exception(f"{model} is not found at {api_base}")
