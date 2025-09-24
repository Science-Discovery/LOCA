import json
import asyncio
from src.api import chat_openai_compatible, chat_gemini_2_5
from typing import Optional, Iterable, Any, cast, Callable, Coroutine, Union, TypeVar
from openai import NOT_GIVEN, NotGiven
from openai.types import ReasoningEffort

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionToolMessageParam
from src.utils import MessageList

# Global semaphore to control concurrent API calls
# Default limit: 10 concurrent calls, can be adjusted as needed
_api_semaphore = asyncio.Semaphore(400)

def _parse_model_and_provider(model: str) -> tuple[str, str]:
    """
    Parse model string to extract provider and model name.
    Format: provider@model_name or just model_name (defaults to openai-comp)
    
    Args:
        model: Model string, either "model_name" or "provider@model_name"
        
    Returns:
        Tuple of (actual_model_name, api_provider)
    """
    if "@" in model:
        provider, actual_model = model.split("@", 1)
        return actual_model, provider
    else:
        return model, "openai-comp"

async def direct_chat(model: str, messages: Union[list[dict[str, Any]], MessageList], record_full_api_path: Optional[str] = None,
                temperature: Optional[float] | NotGiven = NOT_GIVEN,
                max_retries: int = 50,
                ) -> str:
    async with _api_semaphore:
        # TODO: create a abstract class for all chat APIs
        # Parse model string to extract provider and actual model name
        actual_model, api_provider = _parse_model_and_provider(model)
        if "gemini" in model:
            gemini_system_instruction = None
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    gemini_system_instruction = msg["content"]
                elif msg["role"] == "user":
                    gemini_messages.append({
                        "role": "user",
                        "parts": [{ "text": msg["content"] }]
                    })
                elif msg["role"] == "assistant":
                    gemini_messages.append({
                        "role": "model",
                        "parts": [{ "text": msg["content"] }]
                    })
                else:
                    raise ValueError(f"Unsupported role: {msg['role']}")
            res = await chat_gemini_2_5(
                model=actual_model,
                system_instruction=gemini_system_instruction,
                messages=gemini_messages,
                record_full_api_path=record_full_api_path,
                temperature=temperature,
                max_retries=max_retries
            )
            return res
        if actual_model.startswith("o"):
            reasoning_effort = "high"
        else:
            reasoning_effort = None
        res = await chat_openai_compatible(
            model=actual_model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            record_full_api_path=record_full_api_path,
            temperature=temperature,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            api_provider=api_provider,
        )
        return res.content or ""


T = TypeVar('T')

async def safe_chat(model: str, messages: Union[list[dict[str, Any]], MessageList], record_full_api_path: Optional[str] = None,
            temperature: Optional[float] | NotGiven = NOT_GIVEN,
            max_retries: int = 3,
            validator: Optional[Callable[[str], bool]] = None,
            formatter: Optional[Callable[[str], T]] = None,
            debug: bool = False
            ) -> Union[T, str]:
    while True:
        res = await direct_chat(
            model, messages, record_full_api_path, temperature, max_retries
        )
        if validator is None or validator(res):
            return res if formatter is None else formatter(res)
        if debug:
            print(f"Warning: Failed to validate response, retrying...", flush=True)
            print(res, flush=True)
