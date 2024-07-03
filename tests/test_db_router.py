from src.generator import Generator
from src.db_router import DBRouter


db_router = DBRouter(Generator().db)
url = "https://www.espn.com/soccer/report/_/gameId/690575"


def test_get_all_documents_metadata():
    metadata = db_router.get_all_documents_metadata
    assert len(metadata) >= 0


def test_add_document_by_url():
    ids = db_router.add_document_by_url(url)
    assert len(ids) >= 0

    metadata = db_router.get_all_documents_metadata
    added_docs = [doc for doc in metadata if doc["source"] == url]

    assert len(added_docs) == 1
    assert added_docs[0]["count"] == 3


def test_delete_documents_by_url():
    assert db_router.delete_documents_by_url(url) == True

    metadata = db_router.get_all_documents_metadata
    deleted_docs = [doc for doc in metadata if doc["source"] == url]
    assert len(deleted_docs) == 0
