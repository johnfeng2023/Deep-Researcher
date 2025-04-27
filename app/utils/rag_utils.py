import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.utils.config import config
from app.utils.document_utils import extract_text_from_file, chunk_document

# Directory to store vector databases
VECTOR_DB_DIR = "data/vector_stores"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for the Deep-Researcher.
    This class handles document ingestion, embedding, storage, and retrieval.
    """
    
    def __init__(self, collection_name: str = "default"):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name of the vector collection to use
        """
        self.collection_name = collection_name
        self.vector_db_path = os.path.join(VECTOR_DB_DIR, collection_name)
        
        # IMPORTANT: Use exactly the same model name as in fix_vector_store.py
        model_name = "all-MiniLM-L6-v2"
        
        # Initialize the embedding model with the exact same parameters
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize or load the vector store
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize or load the vector store."""
        if os.path.exists(self.vector_db_path) and os.path.isdir(self.vector_db_path):
            # Load existing vector store
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_db_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing vector store from {self.vector_db_path}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self._create_new_vector_store()
        else:
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        """Create a new vector store."""
        # Get the embedding dimensions without using .shape
        test_embedding = self.embeddings.embed_query("test")
        embedding_dim = len(test_embedding) if isinstance(test_embedding, list) else test_embedding.shape[0]
        print("Creating new vector store with embedding dimension:", embedding_dim)
        
        # Create an empty document with a simple test text
        self.vector_store = FAISS.from_documents(
            documents=[Document(page_content="Initialization document", metadata={})],
            embedding=self.embeddings
        )
        self.vector_store.save_local(self.vector_db_path)
        print(f"Created new vector store at {self.vector_db_path}")
    
    def add_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Process a document and add its content to the vector store."""
        if metadata is None:
            metadata = {}
        
        try:
            # Extract text from the document
            print(f"Extracting text from: {file_path}")
            text = extract_text_from_file(file_path)
            print(f"Extracted text length: {len(text)}")
            print(f"Text preview: {text[:100]}...")  # Print a preview
            
            # Add file information to metadata
            file_metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1].lower(),
                **metadata
            }
            
            # Chunk the document
            chunks = chunk_document(text)
            print(f"Created {len(chunks)} chunks")
            
            if not chunks:
                print("WARNING: No chunks created. Text might be too short.")
                # Create at least one chunk with the available text
                if text.strip():
                    chunks = [text]
                    print("Created a single chunk with all text")
            
            # Create Document objects with metadata
            documents = [
                Document(page_content=chunk, metadata=file_metadata)
                for chunk in chunks
            ]
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Save updated vector store
            self.vector_store.save_local(self.vector_db_path)
            
            return len(chunks)
        except Exception as e:
            import traceback
            print(f"Error processing document: {e}")
            print(traceback.format_exc())  # Print full stack trace
            return 0
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add arbitrary text to the vector store.
        
        Args:
            text: Text content to add
            metadata: Metadata to store with the text
            
        Returns:
            Number of chunks added to the vector store
        """
        if metadata is None:
            metadata = {}
        
        # Chunk the text
        chunks = chunk_document(text)
        
        # Create Document objects with metadata
        documents = [
            Document(page_content=chunk, metadata=metadata)
            for chunk in chunks
        ]
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Save updated vector store
        self.vector_store.save_local(self.vector_db_path)
        
        return len(chunks)
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        print(f"Query: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        print(f"Found {len(docs)} relevant documents")
        for i, doc in enumerate(docs):
            print(f"Doc {i} source: {doc.metadata.get('source', 'unknown')}")
        return docs
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve documents relevant to a query with similarity scores.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of tuples containing document and similarity score
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def augment_query(self, query: str, k: int = 3) -> str:
        """
        Augment a query with relevant context from retrieved documents.
        
        Args:
            query: The original query
            k: Number of documents to retrieve for context
            
        Returns:
            Augmented query with context
        """
        # Retrieve relevant documents
        documents = self.retrieve(query, k=k)
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Augment the query with context
        augmented_query = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {query}"""
        
        return augmented_query
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current vector collection.
        
        Returns:
            Dictionary with collection information
        """
        # Get vector store index and check how many vectors it contains
        try:
            index = self.vector_store.index
            vector_count = len(index.index_to_docstore_id)
        except:
            vector_count = 0
        
        return {
            "collection_name": self.collection_name,
            "vector_count": vector_count,
            "vector_db_path": self.vector_db_path,
            "embedding_model": self.embeddings.model_name
        } 