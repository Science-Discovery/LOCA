import os
import json
import asyncio
from anthropic import AsyncAnthropic, NOT_GIVEN

client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)

_CONFIG = {
    "claude-3-7-sonnet-20250219": {
        "max_tokens": 128000,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 63999,
        },
        "extra_headers": {
            "anthropic-beta": "output-128k-2025-02-19"
        }
    },
    "claude-3-5-sonnet-20241022": {
        "max_tokens": 128000,
    }
}

async def chat_anthropic(
    model: str,
    messages: dict,
    record_full_api_path: str = None,
    temperature: float = NOT_GIVEN,
    max_retries=3,
    debug_mode=False,
) -> str:
    CONFIG_MODEL = _CONFIG.get(model)
    if CONFIG_MODEL is None:
        raise ValueError(f"Model {model} not supported. Supported models are: {list(_CONFIG.keys())}")
    for attempt in range(max_retries):
        try:
            completion = await client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=CONFIG_MODEL["max_tokens"],
                thinking=CONFIG_MODEL.get("thinking", NOT_GIVEN),
                extra_headers=CONFIG_MODEL.get("extra_headers", NOT_GIVEN),
                stream=True,
            )

            full_output = []
            result = ""
            last_block = None
            async for message in completion:
                if debug_mode:
                    if message.type == "content_block_start":
                        last_block = message.content_block.type
                        if last_block == "thinking":
                            print("<thinking>", end="\n", flush=True)
                    if message.type == "content_block_stop":
                        if last_block == "thinking":
                            print("\n</thinking>", end="\n", flush=True)
                            last_block = None
                        if last_block == "text":
                            print("", end="\n", flush=True)
                            last_block = None
                if message.type == "content_block_delta":
                    if debug_mode:
                        if message.delta.type == "thinking_delta":
                            print(message.delta.thinking, end="", flush=True)
                        if message.delta.type == "text_delta":
                            print(message.delta.text, end="", flush=True)
                    if message.delta.type == "text_delta":
                        result = result + message.delta.text
                full_output.append(message.to_dict())

            if record_full_api_path:
                if not record_full_api_path.endswith(".json"):
                    raise ValueError("record_full_api_path must end with .json")
                with open(record_full_api_path.replace(".json", ".input.json"), "w") as f:
                    json.dump(messages, f, indent=4)
                with open(record_full_api_path, "w") as f:
                    json.dump(full_output, f, indent=4)
            return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
