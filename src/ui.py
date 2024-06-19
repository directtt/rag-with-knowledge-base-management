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
        if self.transcription:
            st.write(f"Transcription: {self.transcription}")
        else:
            st.write("Error transcribing audio.")

    def get_user_input(self):
        return st.text_input(
            "", value=self.transcription if self.transcription else "", key="input"
        )

    @staticmethod
    def display_conversation(history):
        eleven_labs = ElevenLabs(api_key=ELEVEN_API_KEY)
        for i in range(len(history["generated"])):
            message(history["past"][i], is_user=True, key=f"{i}_user")
            message(history["generated"][i], key=f"{i}")
            # TODO: later change API key
            # audio = eleven_labs.generate(text=history["generated"][i], stream=False)
            #  = b"".join(audio)
            # st.audio(audio_bytes, format="audio/mp3")

    def main(self):
        st.title("rag-with-voice-assistant 🌐")

        self.record_and_transcribe_audio()
        user_input = self.get_user_input()

        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]

        if user_input:
            output = self.generator.search_db(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output["result"])

        if st.session_state["generated"]:
            self.display_conversation(st.session_state)


if __name__ == "__main__":
    ui = UI(generator=Generator())
    ui.main()
