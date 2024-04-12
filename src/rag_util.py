import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer


CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)
ACCESS_TOKEN = os.getenv(
    "ACCESS_TOKEN"
)  # reads .env file with ACCESS_TOKEN=<your hugging face access token>


class Encoder:
    def __init__(
        # model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
        self, model_name: str = "google/gemma-2b-it", device="cuda"
    ):

        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            # token=ACCESS_TOKEN,
            model_kwargs={"device": device},
        )


class FaissDb:
    def __init__(self, docs, embedding_function):
        self.db = FAISS.from_documents(
            docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        )
        print("=======================================================================")
        print(self.db)
        print("=======================================================================")
        # print(docs)
        print("=======================================================================")

    def similarity_search(self, question: str, k: int = 3):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context


def load_and_split_pdfs(file_paths: list, chunk_size: int = 128):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            # "sentence-transformers/all-MiniLM-L12-v2"
            "google/gemma-2b-it",
            token=ACCESS_TOKEN
        ),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        strip_whitespace=True,
    )
    docs = text_splitter.split_documents(pages)
    return docs
