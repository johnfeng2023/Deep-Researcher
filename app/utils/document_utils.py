import os
import tempfile
from typing import List, Dict, Any, Optional, BinaryIO, Tuple
import fitz  # PyMuPDF
import docx
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def save_uploaded_file(uploaded_file: BinaryIO, upload_dir: str = "data/uploads") -> str:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        uploaded_file: The file object from Streamlit file_uploader
        upload_dir: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create a temporary file to save the upload
    with tempfile.NamedTemporaryFile(delete=False, dir=upload_dir, suffix='_'+uploaded_file.name) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text content from a document file.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Extracted text content
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension in ['.txt', '.md', '.csv']:
        return extract_text_from_text(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)

def extract_text_from_text(file_path: str) -> str:
    """Extract text from a plain text file."""
    loader = TextLoader(file_path)
    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)

def chunk_document(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split a document into manageable chunks.
    
    Args:
        text: The document text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def process_uploaded_documents(uploaded_files: List[BinaryIO]) -> str:
    """
    Process multiple uploaded documents and combine their content.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        
    Returns:
        Combined document content
    """
    all_text = []
    
    for uploaded_file in uploaded_files:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)
        
        try:
            # Extract text from the file
            text = extract_text_from_file(file_path)
            all_text.append(f"--- Document: {uploaded_file.name} ---\n{text}")
        except Exception as e:
            all_text.append(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return "\n\n".join(all_text) 