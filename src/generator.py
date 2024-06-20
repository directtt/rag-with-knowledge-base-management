import openai
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
import streamlit as st

from consts import ACTIVELOOP_DATASET_PATH


class Generator:
    """
    A class to generate text & audio using OpenAI's models.
    """

    def __init__(
        self,
        chat_model_name: str = "gpt-3.5-turbo",
        transcription_model_name: str = "whisper-1",
        dataset_path: str = ACTIVELOOP_DATASET_PATH,
    ):
        self.chat_model_name = chat_model_name
        self.transcription_model_name = transcription_model_name

        self.db = self._load_embeddings_and_database(dataset_path)
        self.chat_model = self._load_chat_model()

    @st.cache_resource
    def _load_embeddings_and_database(_self, dataset_path: str) -> DeepLake:
        try:
            embeddings = OpenAIEmbeddings()
            db = DeepLake(
                dataset_path=dataset_path, read_only=True, embedding_function=embeddings
            )
            return db
        except Exception as e:
            raise Exception(f"Error loading embeddings and database: {str(e)}")

    @st.cache_resource
    def _load_chat_model(
        _self, distance_metric="cos", fetch_k: int = 100, k: int = 4
    ) -> RetrievalQA:
        try:
            retriever = _self.db.as_retriever()
            retriever.search_kwargs = {
                "distance_metric": distance_metric,
                "fetch_k": fetch_k,
                "k": k,
            }
            chat_model = RetrievalQA.from_llm(
                ChatOpenAI(model_name=_self.chat_model_name),
                retriever=retriever,
                return_source_documents=True,
            )
            return chat_model
        except Exception as e:
            raise Exception(f"Error loading chat model: {str(e)}")

    def search_db(self, user_input: str) -> str:
        """
        Invoke the chat model to search the database using retrieval-based QA model.

        Args:
            user_input: The user's input to search the database with.

        Returns:
            The response from the chat model.
        """
        try:
            return self.chat_model.invoke({"query": user_input})
        except Exception as e:
            raise Exception(f"Error searching database: {str(e)}")

    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe the audio file using the Whisper API.

        Args:
            audio_file_path: The path to the audio file to transcribe.

        Returns:
            The transcription of the audio file.
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model=self.transcription_model_name,
                    file=audio_file,
                )
            return response.text
        except Exception as e:
            raise Exception(f"Error calling Whisper API: {str(e)}")
