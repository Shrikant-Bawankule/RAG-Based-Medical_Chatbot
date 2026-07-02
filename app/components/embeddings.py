import os
import ssl
from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Intializing our Huggingface embedding model")

        # Bypass SSL verification for environments with corporate SSL inspection
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        ssl._create_default_https_context = ssl._create_unverified_context

        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        logger.info("Huggingface embedding model loaded sucesfully....")

        return model
    
    except Exception as e:
        error_message=CustomException("Error occured while loading embedding model" , e)
        logger.error(str(error_message))
        raise error_message