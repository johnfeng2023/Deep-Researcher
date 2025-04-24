import streamlit as st
from typing import Dict, Any, Optional

from app.utils.config import config

def render_sidebar() -> Dict[str, bool]:
    """
    Render the sidebar with search configuration options.
    
    Returns:
        Dictionary with search configuration settings
    """
    with st.sidebar:
        st.title("Research Options")
        
        st.subheader("Search Sources")
        
        # Search options
        web_search = st.checkbox("Web Search", value=config.search_config.web_search_enabled, 
                               help="Enable web search using search engines")
        
        # Web search engine options
        if web_search:
            st.write("Web Search Engines:")
            col1, col2 = st.columns(2)
            
            with col1:
                serpapi_search = st.checkbox("SerpAPI", value=config.search_config.web_search_config.use_serpapi,
                                          help="Use SerpAPI for web search")
                tavily_search = st.checkbox("Tavily", value=config.search_config.web_search_config.use_tavily,
                                         help="Use Tavily search API")
            
            with col2:
                duckduckgo_search = st.checkbox("DuckDuckGo", value=config.search_config.web_search_config.use_duckduckgo,
                                             help="Use DuckDuckGo for web search")
            
            # API configuration warnings for web search
            if serpapi_search and not config.is_api_configured("serpapi"):
                st.warning("⚠️ SerpAPI key not configured. SerpAPI search may not work.")
            
            if tavily_search and not config.is_api_configured("tavily"):
                st.warning("⚠️ Tavily API key not configured. Tavily search may not work.")
        
        st.write("---")
        st.subheader("Academic Sources")
        
        arxiv_search = st.checkbox("arXiv Papers", value=config.search_config.arxiv_search_enabled,
                                 help="Search for research papers on arXiv")
        
        scholar_search = st.checkbox("Google Scholar", value=config.search_config.scholar_search_enabled,
                                   help="Search for academic papers on Google Scholar")
        
        st.write("---")
        st.subheader("Multimedia Sources")
        
        youtube_search = st.checkbox("YouTube Videos", value=config.search_config.youtube_search_enabled,
                                   help="Search for relevant videos on YouTube")
        
        st.write("---")
        st.subheader("Social Media Sources")
        
        twitter_search = st.checkbox("Twitter", value=config.search_config.twitter_search_enabled,
                                   help="Search for related tweets")
        
        linkedin_search = st.checkbox("LinkedIn", value=config.search_config.linkedin_search_enabled,
                                    help="Search for related LinkedIn posts")
        
        # API configuration warnings for social media
        st.write("---")
        st.subheader("API Configuration Status")
        
        if twitter_search and not config.is_api_configured("twitter"):
            st.warning("⚠️ Twitter API not configured. Twitter search may not work.")
        
        if linkedin_search and not config.is_api_configured("linkedin"):
            st.warning("⚠️ LinkedIn API not configured. LinkedIn search may not work.")
        
        # About section
        st.write("---")
        st.markdown("### About")
        st.markdown("""
        **Deep Researcher** helps you conduct comprehensive research on any topic by leveraging multiple data sources.
        
        Built with:
        - LangChain & LangGraph
        - Ollama (gemma3:1b)
        - Streamlit
        
        [View on GitHub](https://github.com/Sallyliubj/Deep-Researcher)
        """)
    
    # Return the search configuration
    return {
        "web_search_enabled": web_search,
        "arxiv_search_enabled": arxiv_search,
        "scholar_search_enabled": scholar_search,
        "youtube_search_enabled": youtube_search,
        "twitter_search_enabled": twitter_search,
        "linkedin_search_enabled": linkedin_search,
        "web_search_config_use_serpapi": serpapi_search if web_search else False,
        "web_search_config_use_duckduckgo": duckduckgo_search if web_search else False,
        "web_search_config_use_tavily": tavily_search if web_search else False
    } 