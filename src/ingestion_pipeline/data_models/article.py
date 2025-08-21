from typing import Optional, Dict, Any
from pydantic import BaseModel

class Article(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    full_text: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    # Additional fields can be added as needed