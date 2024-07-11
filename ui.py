import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_chat import message

from src.auth import Auth
from src.generator import Generator
from src.db_router import DBRouter
from src.consts import TEMP_AUDIO_PATH, AUDIO_FORMAT

st.set_page_config(page_icon="🌐️")


class UI:
    """
    A class to handle the Streamlit user interface for the application.
    """

    def __init__(self, generator: Generator, db_router: DBRouter):
        self.generator = generator
        self.db_router = db_router

    def _record_and_transcribe_audio(self):
        """
        Record audio from the user and transcribe it using the Whisper API.

        Returns:
            None, saves the audio to a temporary file.
        """
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format=AUDIO_FORMAT)
            with open(TEMP_AUDIO_PATH, "wb") as f:
                f.write(audio_bytes)

            if st.button("Transcribe"):
                st.session_state.transcription = self.generator.transcribe_audio(
                    TEMP_AUDIO_PATH
                )
                os.remove(TEMP_AUDIO_PATH)

    def _get_user_input(self) -> str:
        """
        Get the user's input from the text input field or the transcribed audio.

        Returns:
            The user's input.
        """
        self._record_and_transcribe_audio()

        st.write("##### Or enter the text below:")
        with st.form(key="user_input_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type your message here:",
                value=st.session_state.get("transcription", ""),
                key="input",
            )
            submitted = st.form_submit_button("Submit")
        if submitted and user_input:
            st.session_state.transcription = ""
            return user_input

    @staticmethod
    def _display_conversation(history: st.session_state):
        """
        Display the conversation history in the UI.

        Args:
            history: The conversation history to display.
        """
        for i, (past, generated, source_documents) in enumerate(
            zip(history["past"], history["generated"], history["source_documents"])
        ):
            message(past, is_user=True, key=f"{i}_user")
            message(generated, key=f"{i}")
            with st.expander("See Resources"):
                for source in source_documents:
                    st.write(f"**Source:** {source.metadata['source']}")
                    st.write(f"**Content:** {source.page_content}")
                    st.write(
                        f"**Relevance to Query:** {round(source.metadata['relevance_score'] * 100, 2)}%"
                    )
                    st.divider()

    def show_main_page(self):
        """
        Display the main page of the application.
        """
        st.title("rag-with-voice-assistant 🌐️")

        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]
        if "source_documents" not in st.session_state:
            st.session_state["source_documents"] = [[]]
        if "transcription" not in st.session_state:
            st.session_state["transcription"] = ""

        self._display_conversation(st.session_state)

        user_input = self._get_user_input()

        if user_input:
            with st.spinner("Searching knowledge base..."):
                output = self.generator.search_db(user_input)

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output["answer"])
            st.session_state["source_documents"].append(output["source_documents"])

            st.experimental_rerun()

    def _add_document_by_url(self):
        """
        Display the UI to add a new document by URL.
        """
        st.write("### Add new document by URL")

        with st.form(key="add_document_form", clear_on_submit=True):
            url = st.text_input("Enter URL to add document", key="add_url")
            submitted = st.form_submit_button("Add Document")

        if submitted and url:
            with st.spinner("Adding document..."):
                self.db_router.add_document_by_url(url)
                st.experimental_rerun()

    def _display_existing_documents_metadata(self):
        """
        Display the metadata of the existing documents in the knowledge base.
        """
        metadata_list = self.db_router.get_all_documents_metadata

        st.write("### Existing Documents Metadata")

        col1, col2, col3, col4 = st.columns((3, 3, 1, 1))
        col1.write("**Source**")
        col2.write("**Title**")
        col3.write("**Count**")
        col4.write("**Action**")

        for i, metadata in enumerate(metadata_list):
            col1, col2, col3, col4 = st.columns((3, 3, 1, 1))
            col1.write(metadata["source"])
            col2.write(metadata.get("title", "No Title"))
            col3.write(metadata["count"])

            if col4.button("Delete", key=f"delete_{i}"):
                with st.spinner("Deleting document..."):
                    self.db_router.delete_documents_by_url(metadata["source"])
                    st.experimental_rerun()
            st.divider()

    def show_knowledge_base_page(self):
        """
        Display the knowledge base management page of the application.
        """
        st.title("knowledge-base-management 📖")

        self._add_document_by_url()

        self._display_existing_documents_metadata()

    def main(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Go to", ("RAG with Voice Assistant", "Knowledge Base Management")
        )

        if page == "RAG with Voice Assistant":
            self.show_main_page()
        elif page == "Knowledge Base Management":
            self.show_knowledge_base_page()


if __name__ == "__main__":
    auth = Auth()
    auth.authentication_widget()

    generator = Generator(st.session_state["credentials"])
    ui = UI(generator, DBRouter(st.session_state["credentials"], generator.db))
    ui.main()
