import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs.client import ElevenLabs
from streamlit_chat import message
from generator import Generator

from consts import TEMP_AUDIO_PATH, AUDIO_FORMAT, ELEVEN_API_KEY


class UI:
    """
    A class to handle the streamlit user interface for the application.
    """

    def __init__(self, generator: Generator):
        self.generator = generator
        self.transcription = None

    def record_and_transcribe_audio(self):
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
                self.transcription = self.generator.transcribe_audio(TEMP_AUDIO_PATH)
                os.remove(TEMP_AUDIO_PATH)
                self.display_transcription()

    def display_transcription(self):
        """
        Display the transcribed audio in the UI.

        Returns:
            None
        """
        if self.transcription:
            st.write(f"Transcription: {self.transcription}")
        else:
            st.write("Error transcribing audio.")

    def get_user_input(self) -> str:
        """
        Get the user's input from the text input field or the transcribed audio.

        Returns:
            The user's input.
        """
        return st.text_input(
            "", value=self.transcription if self.transcription else "", key="input"
        )

    @staticmethod
    def display_conversation(history: st.session_state):
        """
        Display the conversation history in the UI.

        Args:
            history: The conversation history to display.

        Returns
            None
        """
        eleven_labs = ElevenLabs(api_key=ELEVEN_API_KEY)
        for i, (past, generated, source_documents) in enumerate(
            zip(history["past"], history["generated"], history["source_documents"])
        ):
            message(past, is_user=True, key=f"{i}_user")
            message(generated, key=f"{i}")
            # TODO: later change API key
            # audio = eleven_labs.generate(text=history["generated"][i], stream=False)
            #  = b"".join(audio)
            # st.audio(audio_bytes, format="audio/mp3")
            with st.expander("See Resources"):
                for source in source_documents:
                    st.write(f"**Source:** {source.metadata['source']}")
                    st.write(f"**Content:** {source.page_content}")
                    st.write(
                        f"**Relevance to Query:** {source.metadata['relevance_score'] * 100}%"
                    )

    def main(self):
        st.title("rag-with-voice-assistant 🌐")

        self.record_and_transcribe_audio()
        user_input = self.get_user_input()

        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]
        if "source_documents" not in st.session_state:
            st.session_state["source_documents"] = [[]]

        if user_input:
            output = self.generator.search_db(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output["result"])
            st.session_state["source_documents"].append(output["source_documents"])

        if st.session_state["generated"]:
            self.display_conversation(st.session_state)


if __name__ == "__main__":
    ui = UI(generator=Generator())
    ui.main()
