import os
from dotenv import load_dotenv

load_dotenv()

ACTIVELOOP_ORG_ID = os.getenv("ACTIVELOOP_ORG_ID")
ACTIVELOOP_DATASET_NAME = "rag_with_knowledge_base_management"
ACTIVELOOP_DATASET_PATH = f"hub://{ACTIVELOOP_ORG_ID}/{ACTIVELOOP_DATASET_NAME}"

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"
