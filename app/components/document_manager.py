import os
import streamlit as st
from typing import List, Dict, Any, Optional

from app.utils.rag_utils import RAGSystem
from app.utils.config import config

def render_document_manager(
    collection_name: str = "default"
) -> None:
    """
    Render the document manager component for RAG functionality.
    
    Args:
        collection_name: Name of the vector collection to use
    """
    st.subheader("Document Manager")
    
    # Initialize RAG system
    rag_system = RAGSystem(collection_name=collection_name)
    
    # Get collection info
    collection_info = rag_system.get_collection_info()
    
    # Display collection info
    st.write(f"Collection: **{collection_info['collection_name']}**")
    st.write(f"Documents: **{collection_info['vector_count']}**")
    
    # Document uploader
    st.write("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents to add to the knowledge base",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md"]
    )
    
    if uploaded_files:
        with st.form("document_upload_form"):
            metadata_description = st.text_input(
                "Description (optional)",
                key="doc_description",
                help="Add a description to these documents"
            )
            
            submitted = st.form_submit_button("Process Documents")
            
            if submitted:
                with st.spinner("Processing documents..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        # Save the uploaded file
                        file_path = os.path.join("data/uploads", uploaded_file.name)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Create metadata
                        metadata = {
                            "description": metadata_description,
                            "filename": uploaded_file.name
                        }
                        
                        # Process the document
                        try:
                            chunks_added = rag_system.add_document(file_path, metadata)
                            total_chunks += chunks_added
                            st.success(f"Added {chunks_added} chunks from {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Update collection info
                    collection_info = rag_system.get_collection_info()
                    st.success(f"Total: Added {total_chunks} chunks to the knowledge base")
    
    # Text input for adding arbitrary text
    st.write("### Add Text")
    with st.form("add_text_form"):
        text_content = st.text_area(
            "Text content",
            height=200,
            help="Enter text to add to the knowledge base"
        )
        
        text_title = st.text_input(
            "Title/Source",
            help="Add a title or source for this text"
        )
        
        text_submitted = st.form_submit_button("Add Text")
        
        if text_submitted and text_content:
            with st.spinner("Processing text..."):
                # Create metadata
                metadata = {
                    "source": text_title or "Custom Text",
                    "type": "custom_text"
                }
                
                # Add the text
                try:
                    chunks_added = rag_system.add_text(text_content, metadata)
                    st.success(f"Added {chunks_added} chunks to the knowledge base")
                    
                    # Update collection info
                    collection_info = rag_system.get_collection_info()
                except Exception as e:
                    st.error(f"Error adding text: {str(e)}") 