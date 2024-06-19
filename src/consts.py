import os
from dotenv import load_dotenv

load_dotenv()

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ACTIVELOOP_ORG_ID = os.getenv("ACTIVELOOP_ORG_ID")
ACTIVELOOP_DATASET_NAME = "langchain_course_jarvis_assistant"  # TODO: change this later
ACTIVELOOP_DATASET_PATH = f"hub://{ACTIVELOOP_ORG_ID}/{ACTIVELOOP_DATASET_NAME}"

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"
