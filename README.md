# rag-with-knowledge-base-management

RAG (Retrieval-Augmented Generation) app integrated with a voice assistant and knowledge base management system.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## Table of Contents
- [Introduction](#introduction)
- [Preview](#preview)
- [Technologies](#technologies)
  - [LangChain](#langchain)
  - [OpenAI Models](#openai-models)
  - [DeepLake Vector Store](#deeplake-vector-store)
  - [Apify](#apify)
  - [Streamlit](#streamlit)
- [Installation & usage](#installation--usage)
- [API keys](#api-keys)
- [License](#license)
- [References](#references)

## Introduction

This application integrates a RAG (Retrieval-Augmented Generation) model with a voice assistant, allowing users to interact with the system via voice or text input.
Additionally, it includes a knowledge base management system, enabling users to add, view, and delete documents used by the RAG model via URLs.

## Preview

https://github.com/directtt/rag-with-knowledge-base-management/assets/72359171/c3ef6984-c06a-4c72-b364-361acffeae01

## Technologies

### LangChain
[LangChain](https://github.com/langchain-ai/langchain) is a framework designed for building applications that leverage language models. It provides tools for connecting language models to external data sources, enabling more complex and contextual interactions.

### OpenAI Models

The application uses several [OpenAI](https://platform.openai.com/) models to provide conversational capabilities and document retrieval:
- Chat Model (**default:** `gpt-3.5-turbo`) to generate responses based on user queries and previous conversation context.
- Whisper API (**default:** `whisper-1`) for automatic speech recognition to transcribe audio inputs from users.

Additionally, [Cohere](https://cohere.com/) Re-ranker (**default:** `rerank-english-v2.0`) to improve the relevance of retrieved documents by re-ranking them based on their relevance to the query.

### DeepLake Vector Store
[DeepLake](https://github.com/activeloopai/deeplake) is used as a vector store to store and retrieve document embeddings. It facilitates efficient similarity search and retrieval of relevant documents from the knowledge base.

### Apify
[Apify](https://apify.com/) is a web scraping and automation platform that allows for the extraction of data from websites. It is used to scrape documents from URLs provided by users and store them in the knowledge base.

### Streamlit
[Streamlit](https://github.com/streamlit/streamlit) is an open-source app framework that allows for the creation of custom web applications for machine learning and data science projects with minimal effort. It is used here to build the user interface of the application.

## Installation & usage

To install the application locally, you need to have [Docker](https://docs.docker.com/get-docker/) installed on your machine.
Then, run following commands:

1. Build the Docker image:
```bash
docker build -t rag-with-knowledge-base-management .
```

2. Run the Docker container:
```bash
docker run -p 8501:8501 rag-with-knowledge-base-management
```

The application should now be accessible at http://localhost:8501.

## API keys
Please make sure to add your API keys to the `.env` file before running the application. The following keys inside `.env.example` need to be filled in:
- `OPENAI_API_KEY` - [OpenAI](https://platform.openai.com/) API key
- `COHERE_API_KEY` - [Cohere](https://cohere.com/) API key
- `APIFY_API_TOKEN` - [Apify](https://apify.com/) API token
- `ACTIVELOOP_TOKEN` - [ActiveLoop](https://activeloop.ai/) API token
- `ACTIVELOOP_ORG_ID` - [ActiveLoop](https://activeloop.ai/) organization ID

## License

Distributed under the open-source Apache 2.0 License. See `LICENSE` for more information.

## References

Following repositories were useful in building this project:
- TODO

