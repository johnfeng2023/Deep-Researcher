import os
import streamlit as st
from typing import Dict, Any, Optional, List

from app.utils.config import config
from app.models.research_agent import run_research_agent
from app.components.sidebar import render_sidebar
from app.components.document_upload import render_document_upload
from app.components.visualization import render_graph_visualization
from app.components.email_sender import render_email_form

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
    An AI-powered research agent that helps you find and synthesize information 
    from multiple sources: web, academic papers, YouTube, social media, and more.
    """
)

# Session state for storing research results
if "research_done" not in st.session_state:
    st.session_state.research_done = False
    
if "research_state" not in st.session_state:
    st.session_state.research_state = {}
    
if "research_result" not in st.session_state:
    st.session_state.research_result = ""
    
if "research_query" not in st.session_state:
    st.session_state.research_query = ""

# Render the sidebar and get search configuration
search_config = render_sidebar()

# Main content
st.markdown("### Enter Your Research Question")

# Research question input form
with st.form("research_form"):
    query = st.text_area(
        "What would you like to research?",
        placeholder="E.g., What are the latest developments in quantum computing and its potential applications?",
        height=100
    )
    
    # Document upload
    document_content = render_document_upload()
    
    # Form submission
    submitted = st.form_submit_button("Start Research")
    
    if submitted and query:
        with st.spinner("Researching... This may take a few minutes."):
            # Adjust query if document content is available
            if document_content:
                enhanced_query = f"{query}\n\nAdditional Context from Uploaded Documents:\n{document_content}"
            else:
                enhanced_query = query
            
            # Run the research agent
            result, state = run_research_agent(enhanced_query, search_config)
            
            # Store results in session state
            st.session_state.research_done = True
            st.session_state.research_state = state
            st.session_state.research_result = result
            st.session_state.research_query = query
            
            # Force a rerun to display results
            st.rerun()
    elif submitted:
        st.error("Please enter a research question.")

# Display results if available
if st.session_state.research_done:
    st.markdown("---")
    st.markdown("## Research Results")
    st.markdown(f"**Question:** {st.session_state.research_query}")
    
    result_container = st.container()
    with result_container:
        st.markdown(st.session_state.research_result)
    
    # Add a download button for results
    st.download_button(
        label="Download Results",
        data=st.session_state.research_result,
        file_name="research_results.txt",
        mime="text/plain"
    )
    
    # Visualization of the research process
    st.markdown("---")
    render_graph_visualization(st.session_state.research_state)
    
    # Email sender form
    st.markdown("---")
    render_email_form(
        research_query=st.session_state.research_query,
        research_result=st.session_state.research_result
    )

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