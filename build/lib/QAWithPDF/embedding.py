from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, documents):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.
    
    Parameters:
    - model: The language model to use with embeddings.
    - documents: The documents to embed in the vector store.

    Returns:
    - query_engine: An index of vector embeddings for efficient similarity queries.
    """

    try:
        logging.info("Initializing Gemini Embedding model...")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        service_context = ServiceContext.from_defaults(llm=model, embed_model=gemini_embed_model)

        logging.info("Creating VectorStoreIndex from documents...")
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()

        logging.info("Setting up query engine...")
        query_engine = index.as_query_engine()
        return query_engine

    except Exception as e:
        logging.error("Error occurred while downloading Gemini embedding.")
        raise customexception(e, sys)
