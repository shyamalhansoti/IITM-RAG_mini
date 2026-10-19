from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
     sys.path.insert(0, str(ROOT_DIR))

class Settings(BaseModel):
    temperature: float=Field(0.0, ge=0.0, le=1.0) 
    COLLECTION_NAME: str = "mp2_sherlock"
    EMBEDDING_MODEL: str   = "text-embedding-3-small"
    EMBEDDING_DIM: int     = 1536
    CHAT_MODEL: str       = "gpt-4o-mini"
    TARGET_CHUNK_SIZE: int = 500   # characters
    CHUNK_OVERLAP: int     = 80    # characters
