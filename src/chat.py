import os
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs.client import ElevenLabs
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

# Load API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")


# Function to load embeddings and database with caching
@st.cache_resource
def load_embeddings_and_database(dataset_path):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=dataset_path, read_only=True, embedding_function=embeddings
    )
    return db


# Function to load the OpenAI model with caching
@st.cache_resource
def load_chat_model(model_name: str = "gpt-3.5-turbo"):
    return ChatOpenAI(model_name=model_name)


# Transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path, openai_key):
    openai.api_key = openai_key
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return response.text
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None


def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)
        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH, openai.api_key)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)
    return transcription


def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
    else:
        st.write("Error transcribing audio.")


def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")


def search_db(user_input, db, model):
    retriever = db.as_retriever()
    retriever.search_kwargs = {"distance_metric": "cos", "fetch_k": 100, "k": 4}
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({"query": user_input})


def display_conversation(history):
    eleven_labs = ElevenLabs(api_key=eleven_api_key)
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=f"{i}_user")
        message(history["generated"][i], key=f"{i}")
        # TODO: later change API key
        # audio = eleven_labs.generate(text=history["generated"][i], stream=False)
        #  = b"".join(audio)
        # st.audio(audio_bytes, format="audio/mp3")


def main():
    st.title("rag-with-voice-assistant 🌐")

    my_activeloop_org_id = "directtt"
    my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = load_embeddings_and_database(dataset_path)
    model = load_chat_model()

    transcription = record_and_transcribe_audio()
    user_input = get_user_input(transcription)

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    if user_input:
        output = search_db(user_input, db, model)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output["result"])

    if st.session_state["generated"]:
        display_conversation(st.session_state)


if __name__ == "__main__":
    main()
