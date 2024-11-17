"""Define all global constants for the project."""
import os

NVCF_API = "<NVCF_API_KEY>"
OPENAI_API_KEY = "<OPENAI_API_KEY>"
URL = "https://integrate.api.nvidia.com/v1"

LOCAL_CACHE = os.getenv("TAO_MM_CACHE", os.path.abspath(os.path.expanduser("~/.cache")))
APP_CACHE = os.path.join(LOCAL_CACHE, "tao_mm_workflows")
