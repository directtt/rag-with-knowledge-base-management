import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_HELP = """
You can sign-up for OpenAI's API [here](https://openai.com/blog/openai-api).\n
Once you are logged in, you find the API keys [here](https://platform.openai.com/account/api-keys)
"""
ACTIVELOOP_HELP = """
You can create an ActiveLoops account (including 500GB of free database storage) [here](https://www.activeloop.ai/).\n
Once you are logged in, you find the API token [here](https://app.activeloop.ai/profile/directtt/apitoken).\n
The organization name is your username, or you can create new organizations [here](https://app.activeloop.ai/organization/new/create)
"""
COHERE_HELP = """
You can sign-up for Cohere's API [here](https://cohere.ai/).\n
Once you are logged in, you find the API keys [here](https://dashboard.cohere.com/api-keys)
"""
APIFY_HELP = """
You can sign-up for Apify's API [here](https://apify.com/).\n
Once you are logged in, you find the API token [here](https://console.apify.com/settings/integrations)
"""

ACTIVELOOP_DATASET_NAME = "rag_with_knowledge_base_management"

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"
