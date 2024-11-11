from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("Initializing Gemini embedding model...")

        # Configure global settings instead of using ServiceContext
        Settings.llm = model
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        # Initialize VectorStoreIndex directly with documents and embedding model
        index = VectorStoreIndex.from_documents(documents=document, embed_model=Settings.embed_model)
        
        # Persist storage context
        index.storage_context.persist()

        logging.info("Embedding model initialized and index persisted.")
        
        # Initialize query engine with the specified LLM
        query_engine = index.as_query_engine(llm=Settings.llm)
        
        return query_engine

    except Exception as e:
        raise customexception(f"Error occurred in {__file__} line number [{sys._getframe().f_lineno}] error message [{str(e)}]", sys)
