# from langchain.document_loaders import ApifyDatasetLoader
import logging

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.utilities import ApifyWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import DeepLake
from dotenv import load_dotenv

load_dotenv()


def scrape_data(url: str) -> list[Document]:
    """
    Scrape data from a given URL.

    Args:
        url: URL to scrape data from.

    Returns:
        List of scraped documents.
    """
    logging.info(f"Scraping data from url: {url}")

    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": url}]},
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=(
                dataset_item["text"] if dataset_item["text"] else "No content available"
            ),
            metadata={
                "source": dataset_item["url"],
                "title": dataset_item["metadata"]["title"],
            },
        ),
    )
    return loader.load()


def split_data(
    docs, chunk_size: int = 1000, chunk_overlap: int = 20, length_function=len
) -> list[Document]:
    """
    Split the scraped data into smaller chunks.

    Args:
        docs: List of documents to split.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.
        length_function: Function to calculate the length of the document.

    Returns:
        List of split documents.
    """
    logging.info("Splitting data into smaller chunks")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split


def embed_data(docs_split: list[Document]):
    """
    Embed the split documents using Cohere embeddings and store them in DeepLake.

    Args:
        docs_split: List of split documents to embed.

    Returns:
        None
    """
    logging.info("Embedding data and storing in DeepLake")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    my_activeloop_org_id = "directtt"
    my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    dbs = DeepLake(
        dataset_path=dataset_path, embedding_function=embeddings
    )
    dbs.add_documents(docs_split)


def main():
    urls = [
        "https://www.espn.com/soccer/report/_/gameId/700703",
        "https://www.espn.com/soccer/report/_/gameId/690579",
        "https://www.espn.com/soccer/report/_/gameId/690591"
    ]
    for url in urls:
        docs = scrape_data(url)
        docs_split = split_data(docs)
        embed_data(docs_split)


if __name__ == "__main__":
    main()
