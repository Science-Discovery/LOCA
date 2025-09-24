import os
import dotenv
import warnings
from .utils import (
    ProblemSet, ProblemSetWithAnswer, ProblemWithAnswer, extract_problem,
    MessageList, load_prompt, bold_stdout
)

__all__ = [
    "load_prompt",
    "ProblemSet",
    "ProblemSetWithAnswer",
    "ProblemWithAnswer",
    "extract_problem",
    "MessageList",
    "ROOT_DIR",
    "RESULT_ROOT_DIR",
]
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_ROOT_DIR = os.path.join(ROOT_DIR, "results")
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"RESULT_ROOT_DIR: {RESULT_ROOT_DIR}")
import sys
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# Load environment variables from .env file
dotenv.load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)
DEBUG_MODE = os.getenv("DEBUG_MODE", default="false").lower() == "true"
if DEBUG_MODE:
    print("Debug mode is enabled.")

# Print available model list
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_BASE_URL and OPENAI_API_KEY:
    print(bold_stdout("OpenAI base URL and API key are set."))
    print(f"OpenAI base URL: {OPENAI_BASE_URL}")
    # print(f"OpenAI API key: {OPENAI_API_KEY}")
else:
    warnings.warn("OpenAI base URL and API key are not set. Please set them in the .env file if you want to use OpenAI API.")

ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY:
    print(bold_stdout("Anthropic base URL and API key are set."))
    print(f"Anthropic base URL: {ANTHROPIC_BASE_URL}")
    # print(f"Anthropic API key: {ANTHROPIC_API_KEY}")
else:
    warnings.warn("Anthropic base URL and API key are not set. Please set them in the .env file if you want to use Anthropic API.")
print("\n", flush=True)



print("\n", flush=True)
print(bold_stdout("All modules imported successfully."), flush=True)
print("\n", flush=True)

os.makedirs(RESULT_ROOT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT_DIR, ".api.origin"), exist_ok=True)
