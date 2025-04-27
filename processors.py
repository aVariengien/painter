# %%
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import base64
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import anthropic
from anthropic.types import Message
import markdown
from file_import import create_document
from concurrent.futures import ThreadPoolExecutor
import json

@dataclass
class Annotation:
    text: str  # Must be included in the chunk text
    type: str  # e.g. "analogy", "example", etc.
    associated_text: str  # e.g. the analogy text
    image: Optional[str] = None  # base64 encoded image
    extras: Optional[Dict[str, Any]] = None

@dataclass
class AnnotatedChunk:
    chunk_text: str
    annotations: List[Annotation]

@dataclass
class AnnotatedDocument:
    raw_content: str
    text_only: str
    display_markdown: str
    image_map: Dict[str, str]
    annotated_chunks: List[AnnotatedChunk]
    def get_display_ready(self, text: str) -> str:
        """Replace image placeholders with actual image content."""
        result = text
        for placeholder, image_content in self.image_map.items():
            result = result.replace(placeholder, image_content)
        return result



def chunk_markdown(markdown_path: str, chunk_size: int = 1000, max_char: int = 20000, start_idx: int = 0) -> AnnotatedDocument:
    # Read markdown content
    doc = create_document(Path(markdown_path).read_text())
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(doc.text_only[start_idx:start_idx+max_char])
    
    # Convert chunks into ChunkAnnotation objects
    chunk_annotations = []
    for i, chunk in enumerate(chunks):
        chunk_annotation = AnnotatedChunk(
            chunk_text=chunk,
            annotations=[]
        )
        chunk_annotations.append(chunk_annotation)
    
    document = AnnotatedDocument(
        raw_content=doc.raw_content,
        text_only=doc.text_only,
        display_markdown=doc.text_only,
        image_map=doc.image_map,
        annotated_chunks=chunk_annotations
    )
    
    return document
