import streamlit as st
from typing import Dict, Any, Optional, List
from streamlit.delta_generator import DeltaGenerator
import streamlit.components.v1 as components
import pandas as pd
import json
import re

from app.utils.visualization import research_state_to_mermaid

def render_graph_visualization(state: Dict[str, Any], container: Optional[DeltaGenerator] = None) -> None:
    """
    Render the LangGraph visualization.
    
    Args:
        state: The research agent state from the graph execution
        container: Optional container to render the visualization in
    """
    target = container or st
    
    # Get the Mermaid diagram code
    mermaid_code = research_state_to_mermaid(state)
    
    # Display as Mermaid diagram
    if mermaid_code:
        target.subheader("🔍 LangGraph Visualization")
        render_mermaid(mermaid_code)
    else:
        target.info("No graph data available for visualization.")
    
    # Show execution steps
    render_execution_steps(state)

def render_mermaid(mermaid_code: str) -> None:
    """
    Render a Mermaid diagram.
    
    Args:
        mermaid_code: Mermaid diagram code as a string
    """
    # Wrap the mermaid code in HTML
    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>
    
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    """
    
    # Render the HTML
    components.html(html, height=400)

def render_execution_steps(state: Dict[str, Any]) -> None:
    """
    Render the execution steps of the agent.
    
    Args:
        state: The research agent state from the graph execution
    """
    search_logs = state.get("search_logs", [])
    
    if search_logs:
        st.subheader("More Details")
        
        # Create tabs for overview and detailed results
        details_tab, overview_tab = st.tabs(["Detailed Results", "Steps Overview"])
        
        with details_tab:
            # Ensure we start with index 1 for display
            for i, log in enumerate(search_logs, 1):
                source = log["source"].capitalize()
                with st.expander(f"Step {i}: {source} Search Results"):
                    # For academic search, show structured results
                    if log["source"] == "academic" and "links" in log:
                        for paper in log.get("links", []):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Get paper URL, trying multiple possible fields
                                paper_url = paper.get('url', paper.get('link', '#'))
                                if paper_url == 'No URL available' or not paper_url:
                                    paper_url = '#'
                                    
                                paper_title = paper.get('title', 'Untitled Paper')
                                
                                # Display title with link
                                st.markdown(f"**[{paper_title}]({paper_url})**")
                                
                                # Add direct link button if URL exists
                                if paper_url != '#':
                                    st.markdown(f"[🔗 Open Paper]({paper_url})")
                                
                                # Authors
                                if "authors" in paper:
                                    st.markdown(f"*Authors:* {paper['authors']}")
                                
                                # Publication date
                                pub_date = paper.get('published', paper.get('year', 'Date unknown'))
                                st.markdown(f"*Published:* {pub_date}")
                            
                            with col2:
                                # Abstract/Summary toggle
                                abstract = paper.get('abstract', paper.get('summary', None))
                                if abstract:
                                    if st.button("📄 Show Abstract", key=f"abstract_{i}_{paper_title[:20]}"):
                                        st.markdown("**Abstract:**")
                                        st.markdown(abstract)
                            
                            st.markdown("---")
                    # For web search results, show structured results with links
                    elif log["source"] == "web":
                        results_text = log["results"]
                        # Remove the header if it exists to avoid duplication
                        results_text = results_text.replace("## Web Search Results\n\n", "")
                        sections = results_text.split("###")[1:] if "###" in results_text else [results_text]
                        
                        for section in sections:
                            if not section.strip():
                                continue
                                
                            lines = section.strip().split("\n")
                            if len(lines) >= 2:
                                title = lines[0].strip()
                                link_line = lines[1].strip()
                                snippet = "\n".join(lines[2:]).strip()
                                
                                # Extract URL from markdown link format
                                url_match = re.search(r'\[(.*?)\]\((.*?)\)', link_line)
                                if url_match:
                                    url = url_match.group(2)
                                    
                                    # Display with title, link button, and snippet
                                    st.markdown(f"**{title}**")
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(snippet)
                                    with col2:
                                        st.markdown(f"[🔗 Visit Site]({url})")
                                    st.markdown("---")
                            else:
                                # If the section doesn't match the expected format, display as is
                                st.markdown(section)
                    else:
                        # For other sources, show formatted results
                        st.markdown(log["results"])
        
        with overview_tab:
            steps_data = []
            for i, log in enumerate(search_logs, 1):
                source = log["source"].capitalize()
                results_text = log["results"]
                num_results = len(log.get("links", [])) if "links" in log else 0
                
                steps_data.append({
                    "Step": i,
                    "Source": source,
                    "Query": log["query"][:50] + ("..." if len(log["query"]) > 50 else ""),
                    "Results Found": f"{num_results} items" if num_results > 0 else "See details",
                })
            
            st.dataframe(
                pd.DataFrame(steps_data),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No execution steps available.") 