from llama_index.core import SimpleDirectoryReader
import sys
import os
from exception import customexception
from logger import logging

def load_data(data):
    """
    Load PDF documents from a specified file or directory.

    Parameters:
    - data: The path to the directory containing PDF files or a Streamlit UploadedFile object.

    Returns:
    - A list of loaded PDF documents.
    """
    try:
        logging.info("data loading started...")

        # Check if data is a Streamlit UploadedFile instance
        if hasattr(data, 'read'):
            # Save the uploaded file temporarily
            temp_dir = "temp_data"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, data.name)

            with open(temp_file_path, "wb") as f:
                f.write(data.read())

            # Load data from the temporary directory
            loader = SimpleDirectoryReader(temp_dir)
            documents = loader.load_data()

            # Clean up by removing the temp directory and file
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
        
        else:
            # If data is a directory path, load files from the directory
            loader = SimpleDirectoryReader(data)
            documents = loader.load_data()

        logging.info("data loading completed...")
        return documents

    except Exception as e:
        logging.info("exception in loading data...")
        raise customexception(e, sys)
