import os
import json
import random
import asyncio
from typing import Optional, Iterable
from openai import AsyncOpenAI, NOT_GIVEN, NotGiven
from openai.types import ReasoningEffort
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageParam, ChatCompletionMessage, ChatCompletionToolChoiceOptionParam

# Default client for backward compatibility
openai_comp_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

def _get_client_for_provider(api_provider: str = "openai-comp") -> AsyncOpenAI:
    """Get the appropriate client based on the API provider."""
    return openai_comp_client


async def chat_openai_compatible(
    model: str,
    messages: Iterable[ChatCompletionMessageParam],
    record_full_api_path: Optional[str] = None,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    max_retries = 3,
    reasoning_effort: Optional[ReasoningEffort] | NotGiven = NOT_GIVEN,
    tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    api_provider: str = "openai-comp",
) -> ChatCompletionMessage:
    if record_full_api_path:
        if not record_full_api_path.endswith(".json"):
            raise ValueError("record_full_api_path must end with .json")
        with open(record_full_api_path.replace(".json", ".input.json"), "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
    
    # Create kwargs dict to handle different model requirements
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    
    # Add tools and tool_choice only if provided
    if not isinstance(tools, NotGiven):
        kwargs["tools"] = tools
        if not isinstance(tool_choice, NotGiven):
            kwargs["tool_choice"] = tool_choice
    
    # Add reasoning_effort only for o-series models
    if model.startswith("o") and not isinstance(reasoning_effort, NotGiven):
        kwargs["reasoning_effort"] = reasoning_effort

    # if "r1" in model or "R1" in model:
    #     kwargs["max_tokens"] = 16384  # Example for DeepSeek-R1 models
    
    # Get the appropriate client based on the provider
    provider_client = _get_client_for_provider(api_provider)
    for attempt in range(max_retries):
        try:
            completion = await provider_client.chat.completions.create(**kwargs)
            
            if record_full_api_path:
                with open(record_full_api_path, "w", encoding="utf-8") as f:
                    json.dump(completion.to_dict(), f, indent=4, ensure_ascii=False)
            
            return completion.choices[0].message
        except Exception as e:
            import traceback
            print(f"Attempt {attempt + 1} failed: {repr(e)}", flush=True)
            traceback.print_exc()

            if attempt < max_retries - 1:
                await asyncio.sleep(10 + random.random() * 20)
            else:
                raise  # Re-raise the exception on the last attempt
