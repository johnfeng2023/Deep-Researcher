import os
import streamlit as st
import sys
from typing import Dict, Any, Optional, List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.config import config
from app.models.research_agent import run_research_agent
from app.models.rag_agent import run_rag_agent
from app.components.sidebar import render_sidebar
from app.components.document_upload import render_document_upload
from app.components.document_manager import render_document_manager
from app.components.visualization import render_graph_visualization
from components.email_sender import render_email_form

# Page configuration
st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🔍 Deep Researcher")
st.markdown(
    """
    An AI-powered research assistant that performs comprehensive research using multiple sources and structured methodologies.
    """
)

# Session state for storing research results
if "research_done" not in st.session_state:
    st.session_state.research_done = False
    
if "research_state" not in st.session_state:
    st.session_state.research_state = None

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = "default"

# Render the sidebar and get search configuration
search_config = render_sidebar()

# Main content
st.markdown("### Enter Your Research Question")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Research Agent", "Document Management"])

with tab1:
    # Research form
    with st.form("research_form"):
        question = st.text_area(
            "What would you like to research?",
            placeholder="E.g., What are the latest developments in quantum computing and its potential applications?",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            use_rag = st.checkbox(
                "Use RAG (Retrieval-Augmented Generation)", 
                value=config.rag_config.enabled,
                help="Enhance the research with knowledge from your document collection"
            )
        
        with col2:
            if use_rag:
                rag_collection = st.text_input(
                    "RAG Collection", 
                    value=st.session_state.rag_collection,
                    help="The name of the vector collection to use"
                )
        
        submitted = st.form_submit_button("Start Research")
        
        if submitted and question:
            # Store the collection name
            if use_rag and rag_collection:
                st.session_state.rag_collection = rag_collection
            
            # Get search configuration from session state
            search_config = {
                key: value 
                for key, value in st.session_state.items() 
                if key.startswith(("web_search_", "academic_search_", "social_media_search_"))
            }
            
            # Run research with progress indicator
            with st.spinner("Researching your question..."):
                if use_rag:
                    # Run RAG agent
                    answer, state = run_rag_agent(
                        question=question,
                        collection_name=st.session_state.rag_collection
                    )
                    
                    # Store state for visualization
                    st.session_state.research_state = state
                    
                    # Display answer
                    st.markdown("## Research Results")
                    st.markdown(answer)
                    
                    # Display retrieved documents
                    st.markdown("## Retrieved Documents")
                    for i, doc in enumerate(state.get("retrieved_docs", []), 1):
                        source = doc.get("metadata", {}).get("source", "Unknown source")
                        st.markdown(f"### {i}. {source}")
                        
                        with st.expander("Show content"):
                            st.markdown(doc.get("content", "No content available"))
                else:
                    # Run regular research agent
                    answer, state = run_research_agent(
                        question=question,
                        search_config=search_config
                    )
                    
                    # Store state for visualization
                    st.session_state.research_state = state
                    
                    # Display answer
                    st.markdown("## Research Results")
                    st.markdown(state.get("final_answer", answer))
    
    # Visualization of the research process
    if st.session_state.research_state:
        st.markdown("---")
        render_graph_visualization(st.session_state.research_state)

with tab2:
    # Document management for RAG
    render_document_manager(collection_name=st.session_state.rag_collection)

# Email sender form (if needed)
if st.session_state.research_state and st.session_state.research_state.get("final_answer"):
    st.markdown("---")
    with st.expander("📧 Email Research Results"):
        render_email_form(st.session_state.research_state.get("final_answer", ""))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
    Built using LangChain, LangGraph, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

# Add custom CSS
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E4175;
    }
    .stButton button {
        background-color: #1E4175;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #1E4175;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    # This will only be executed when the script is run directly, not when imported
    pass 