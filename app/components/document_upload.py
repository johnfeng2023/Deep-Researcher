import streamlit as st
from typing import List, Optional
import pandas as pd

from app.utils.document_utils import process_uploaded_documents

def render_document_upload() -> Optional[str]:
    """
    Render document upload section and process uploaded documents.
    
    Returns:
        Document content if files were uploaded, otherwise None
    """
    with st.expander("📄 Upload Documents", expanded=False):
        st.write("Upload documents to include in your research (optional)")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md", "csv"],
            help="Upload documents to be included as context for your research question."
        )
        
        if uploaded_files:
            file_info = []
            for file in uploaded_files:
                file_info.append({
                    "Filename": file.name,
                    "Size": f"{round(file.size / 1024, 1)} KB",
                    "Type": file.type
                })
            
            st.dataframe(
                pd.DataFrame(file_info),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    document_content = process_uploaded_documents(uploaded_files)
                    st.success(f"Successfully processed {len(uploaded_files)} document(s)")
                    
                    with st.expander("View Document Content", expanded=False):
                        st.text_area(
                            "Extracted Content",
                            document_content,
                            height=300,
                            disabled=True
                        )
                    
                    return document_content
    
    return None 