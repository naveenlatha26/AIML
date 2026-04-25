from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class RAG:
    def __init__(self):
        self.pdf_folder_path = os.getenv("SOURCE_DATA")
        self.emb_model_path = os.getenv("EMBED_MODEL")
        self.vector_store_path = os.getenv("VECTOR_STORE")
        self.emb_model = self.get_embedding_model(self.emb_model_path)

    def load_docs(self, path):
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        print("Loaded docs:", len(docs))
        return docs

    def get_embedding_model(self, emb_model):
        return HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def split_docs(self, docs):
        splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(docs)
        print("Split docs:", len(docs))
        return docs

    def populate_vector_db(self):
        docs = self.load_docs(self.pdf_folder_path)
        chunks = self.split_docs(docs)

        db = Chroma.from_documents(
            chunks,
            embedding=self.emb_model,
            persist_directory=self.vector_store_path,
        )
        db.persist()

    def load_vector_db(self):
        if not os.path.exists(self.vector_store_path):
            print("Creating vector DB...")
            self.populate_vector_db()

        return Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.emb_model,
        )

    def get_retriever(self):
        return self.load_vector_db().as_retriever(
            search_kwargs={"k": 3}
        )