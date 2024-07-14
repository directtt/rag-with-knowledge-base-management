import streamlit as st
import openai
import cohere
from apify_client import ApifyClient
import os
import deeplake
from src.consts import (
    OPENAI_HELP,
    ACTIVELOOP_HELP,
    COHERE_HELP,
    APIFY_HELP,
    ACTIVELOOP_DATASET_NAME,
)


class Auth:
    """
    A class to handle the authentication for the application.
    """

    def __init__(self):
        self._init_session_state()

    @staticmethod
    def _init_session_state():
        # Initialize all session state variables with defaults
        SESSION_DEFAULTS = {
            "openai_api_key": "",
            "activeloop_token": "",
            "activeloop_org_id": "",
            "apify_api_token": "",
            "cohere_api_key": "",
            "credentials": {},
            "auth_ok": False,
        }

        for k, v in SESSION_DEFAULTS.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def authentication_widget(self) -> None:
        """
        Display the authentication widget in the Streamlit sidebar.
        """
        with st.sidebar:
            with st.expander(
                "Authentication", expanded=not st.session_state["auth_ok"]
            ), st.form("authentication"):
                st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help=OPENAI_HELP,
                    placeholder="This field is mandatory",
                    key="openai_api_key",
                )
                st.text_input(
                    "ActiveLoop Token",
                    type="password",
                    help=ACTIVELOOP_HELP,
                    placeholder="This field is mandatory",
                    key="activeloop_token",
                )
                st.text_input(
                    "ActiveLoop Organization ID",
                    type="password",
                    help=ACTIVELOOP_HELP,
                    placeholder="This field is mandatory",
                    key="activeloop_org_id",
                )
                st.text_input(
                    "Cohere API Key",
                    type="password",
                    help=COHERE_HELP,
                    placeholder="This field is mandatory",
                    key="cohere_api_key",
                )
                st.text_input(
                    "Apify API Token",
                    type="password",
                    help=APIFY_HELP,
                    placeholder="This field is mandatory",
                    key="apify_api_token",
                )
                submitted = st.form_submit_button("Submit")
                if submitted:
                    self._authenticate()

        if not st.session_state["auth_ok"]:
            st.info(
                "Please enter your credentials or submit to use the default environment variables.",
                icon=":material/info:",
            )
            st.stop()

    @staticmethod
    def _authenticate() -> None:
        # Validate all credentials are set and correct
        openai_api_key = st.session_state["openai_api_key"] or os.environ.get(
            "OPENAI_API_KEY"
        )
        activeloop_token = st.session_state["activeloop_token"] or os.environ.get(
            "ACTIVELOOP_TOKEN"
        )
        activeloop_org_id = st.session_state["activeloop_org_id"] or os.environ.get(
            "ACTIVELOOP_ORG_ID"
        )
        cohere_api_key = st.session_state["cohere_api_key"] or os.environ.get(
            "COHERE_API_KEY"
        )
        apify_api_token = st.session_state["apify_api_token"] or os.environ.get(
            "APIFY_API_TOKEN"
        )

        if not (
            openai_api_key
            and activeloop_token
            and activeloop_org_id
            and cohere_api_key
            and apify_api_token
        ):
            st.session_state["auth_ok"] = False
            st.error("Credentials neither set nor stored", icon=":material/error:")
            return
        try:
            # Try to access the APIs with the provided credentials
            with st.spinner("Authenticating..."):
                openai.api_key = openai_api_key
                openai.models.list()

                deeplake.exists(
                    f"hub://{activeloop_org_id}/{ACTIVELOOP_DATASET_NAME}",
                    token=activeloop_token,
                )

                co = cohere.Client(cohere_api_key)
                co.models.list()

                client = ApifyClient(apify_api_token)
                client.user().get()

        except Exception as e:
            st.session_state["auth_ok"] = False
            st.error(f"Authentication failed: {e}", icon=":material/error:")
            return

        # Store credentials in the session state
        st.session_state["auth_ok"] = True
        st.session_state["credentials"] = {
            "openai_api_key": openai_api_key,
            "activeloop_token": activeloop_token,
            "activeloop_org_id": activeloop_org_id,
            "cohere_api_key": cohere_api_key,
            "apify_api_token": apify_api_token,
        }
        st.success("Authentication successful!", icon=":material/check_circle:")
