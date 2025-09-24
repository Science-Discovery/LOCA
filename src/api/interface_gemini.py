import os
import json
import aiohttp
import asyncio

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_BASE_URL or not GEMINI_API_KEY:
    raise ValueError("GEMINI_BASE_URL and GEMINI_API_KEY must be set in environment variables.")

async def chat_gemini_2_5(
    model,
    messages,
    system_instruction: str = None,
    record_full_api_path: str = None,
    temperature: float = 0,
    max_retries: int = 3
) -> str:
    if model not in ["gemini-2.5-pro-preview-05-06", "gemini-2.5-pro-preview-06-05", "gemini-2.5-pro",
                     "gemini-2.5-flash"]:
        raise ValueError("Unsupported model for Gemini 2.5 API")
    
    url = f"{GEMINI_BASE_URL}/api/google/v1beta/models/{model}:generateContent"
    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": messages,
        "generationConfig": {
            "temperature": temperature,
            "thinkingConfig": {
                "thinkingBudget": 24576
            }
        }
    } if "flash" not in model else {
        "contents": messages,
        "generationConfig": {
            "temperature": temperature,
        }
    }
    if system_instruction is not None:
        data["systemInstruction"] = {
            "parts": [{ "text": system_instruction }]
        }
    # Save input
    if record_full_api_path is not None:
        with open(record_full_api_path.replace(".json", ".input.json"), "w") as f:
            json.dump(messages, f, indent=4)
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=10000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Make API call
                async with session.post(url, params=params, headers=headers, json=data) as response:
                    if response.status != 200:
                        raise Exception(f"API returned status code {response.status}: {await response.text()}")
                    
                    response_data = await response.json()
                    
                    # Save output
                    if record_full_api_path is not None:
                        with open(record_full_api_path, "w") as f:
                            json.dump(response_data, f, indent=4)

                    response_content = response_data["candidates"][0]["content"]
                    if response_content["role"] != "model":
                        raise Exception(f"API returned unexpected role: {response_content['role']}")
                    
                    return "\n".join([msg["text"] for msg in response_content["parts"]])
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {repr(e)}", flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                raise  # Re-raise the exception on the last attempt

