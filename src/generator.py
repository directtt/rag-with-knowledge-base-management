import openai
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
import streamlit as st

from src.consts import ACTIVELOOP_DATASET_NAME


class Generator:
    """
    A class to generate text & audio using OpenAI's models.
    """

    def __init__(
        self,
        credentials: dict[str, str],
        chat_model_name: str = "gpt-3.5-turbo",
        cohere_rerank_model_name: str = "rerank-english-v2.0",
        transcription_model_name: str = "whisper-1",
    ):
        self.credentials = credentials

        self.chat_model_name = chat_model_name
        self.cohere_rerank_model_name = cohere_rerank_model_name
        self.transcription_model_name = transcription_model_name

        self.db = self._load_embeddings_and_database()
        self.chat_model, self.memory = self._load_chat_model()

    @st.cache_resource
    def _load_embeddings_and_database(_self) -> DeepLake:
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=_self.credentials["openai_api_key"]
            )
            ACTIVELOOP_ORG_ID = _self.credentials["activeloop_org_id"]
            db = DeepLake(
                dataset_path=f"hub://{ACTIVELOOP_ORG_ID}/{ACTIVELOOP_DATASET_NAME}",
                embedding_function=embeddings,
                token=_self.credentials["activeloop_token"],
            )
            return db
        except Exception as e:
            raise Exception(f"Error loading embeddings and database: {str(e)}")

    @st.cache_resource
    def _load_chat_model(
        _self, top_n: int = 3, k_history: int = 3
    ) -> tuple[ConversationalRetrievalChain, ConversationBufferWindowMemory]:
        try:
            retriever = _self.db.as_retriever()
            compressor = CohereRerank(
                model=_self.cohere_rerank_model_name,
                top_n=top_n,
                cohere_api_key=_self.credentials["cohere_api_key"],
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            memory = ConversationBufferWindowMemory(
                k=k_history,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            chat_model = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    model_name=_self.chat_model_name,
                    openai_api_key=_self.credentials["openai_api_key"],
                ),
                retriever=compression_retriever,
                memory=memory,
                verbose=True,
                chain_type="stuff",
                return_source_documents=True,
            )
            return chat_model, memory
        except Exception as e:
            raise Exception(f"Error loading chat model: {str(e)}")

    def search_db(self, user_input: str) -> dict[str, any]:
        """
        Invoke the chat model to search the database using retrieval-based QA model.

        Args:
            user_input: The user's input to search the database with.

        Returns:
            The response from the chat model.
        """
        try:
            return self.chat_model(
                {
                    "question": user_input,
                    "chat_history": self.memory.load_memory_variables({}),
                }
            )
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
