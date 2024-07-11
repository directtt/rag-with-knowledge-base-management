import logging
from collections import defaultdict
from langchain_community.utilities import ApifyWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import DeepLake
from dotenv import load_dotenv

load_dotenv()


class DBRouter:
    """
    Router to manage vector database (DeepLake) operations.
    """

    def __init__(self, credentials: dict[str, str], db: DeepLake):
        self.credentials = credentials

        self.db = db
        self.ds = db.ds()

    @property
    def get_all_documents_metadata(self) -> list[dict]:
        """
        Get all documents metadata from the database.

        Returns:
            List of documents metadata.
        """
        try:
            raw_metadata = [doc.data()["value"] for doc in self.ds.metadata]
            return self._parse_metadata(raw_metadata)
        except Exception as e:
            raise Exception(f"Error getting all documents metadata: {str(e)}")

    def delete_documents_by_url(self, url: str) -> bool:
        """
        Delete documents by URL.

        Args:
            url: URL of the documents to delete.

        Returns:
            True if documents were deleted, False otherwise.
        """
        try:
            return self.db.delete(filter={"metadata": {"source": url}})
        except Exception as e:
            raise Exception(f"Error deleting documents by URL: {str(e)}")

    def add_document_by_url(self, url: str) -> list[str]:
        """
        Add a document to vector store by URL.

        Args:
            url: URL of the document to add.

        Returns:
            List of added document IDs.
        """
        try:
            return self.db.add_documents(self._split_data(self._scrape_data(url)))
        except Exception as e:
            raise Exception(f"Error adding document by URL: {str(e)}")

    @staticmethod
    def _parse_metadata(raw_metadata: list[dict]) -> list[dict]:
        metadata_count = defaultdict(int)

        for metadata in raw_metadata:
            metadata_tuple = tuple(metadata.items())
            metadata_count[metadata_tuple] += 1

        unique_metadata_with_counts = [
            {**dict(metadata_tuple), "count": count}
            for metadata_tuple, count in metadata_count.items()
        ]

        return unique_metadata_with_counts

    def _scrape_data(self, url: str) -> list[Document]:
        """
        Scrape data from a given URL.

        Args:
            url: URL to scrape data from.

        Returns:
            List of scraped documents.
        """
        logging.info(f"Scraping data from url: {url}")

        apify = ApifyWrapper(apify_api_token=self.credentials["apify_api_token"])
        loader = apify.call_actor(
            actor_id="apify/website-content-crawler",
            run_input={"startUrls": [{"url": url}]},
            dataset_mapping_function=lambda dataset_item: Document(
                page_content=(
                    dataset_item["text"]
                    if dataset_item["text"]
                    else "No content available"
                ),
                metadata={
                    "source": dataset_item["url"],
                    "title": dataset_item["metadata"]["title"],
                },
            ),
        )
        return loader.load()

    @staticmethod
    def _split_data(
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
