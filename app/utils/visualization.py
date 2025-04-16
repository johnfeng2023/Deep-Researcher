from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import pandas as pd
import json

def get_graph_dict_for_visualization(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert graph state to a format suitable for visualization in Streamlit.
    
    Args:
        graph_state: The graph state from the LangGraph execution
        
    Returns:
        Dictionary with nodes and edges for visualization
    """
    # Extract information from the graph state
    edges = []
    for i, (from_node, metadata) in enumerate(graph_state.get("intermediate_steps", [])):
        # If we have more steps, create an edge
        if i + 1 < len(graph_state.get("intermediate_steps", [])):
            to_node = graph_state["intermediate_steps"][i + 1][0]
            edges.append({
                "from": from_node,
                "to": to_node,
                "metadata": {"transition": i + 1}
            })
    
    # Create a list of nodes
    nodes = []
    for node_name, node_data in graph_state.get("nodes", {}).items():
        node = {
            "id": node_name,
            "label": node_name.capitalize(),
            "type": node_data.get("type", "process"),
        }
        nodes.append(node)
    
    # Handle case for older format of graph state
    if not nodes and "intermediate_steps" in graph_state:
        unique_nodes = set()
        for from_node, _ in graph_state["intermediate_steps"]:
            unique_nodes.add(from_node)
        
        nodes = [{"id": node, "label": node.capitalize(), "type": "process"} for node in unique_nodes]
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def graph_to_mermaid(graph_dict: Dict[str, Any]) -> str:
    """
    Convert a graph dictionary to Mermaid diagram code.
    
    Args:
        graph_dict: Dictionary with nodes and edges
        
    Returns:
        Mermaid diagram code as a string
    """
    mermaid_code = ["graph TD;"]
    
    # Add subgraph for search layer
    search_nodes = [n for n in graph_dict["nodes"] if n["id"].endswith("_search")]
    if search_nodes:
        mermaid_code.append("    subgraph Search_Layer")
        for node in search_nodes:
            mermaid_code.append(f"        {node['id']}[\"{node['label']}\"];")
        mermaid_code.append("    end")
    
    # Add other nodes
    other_nodes = [n for n in graph_dict["nodes"] if not n["id"].endswith("_search")]
    for node in other_nodes:
        mermaid_code.append(f"    {node['id']}[\"{node['label']}\"];")
    
    # Add node styles
    style_defs = {
        "process": "    style {id} fill:#bbdefb,stroke:#1976d2,color:#333;",
        "start": "    style {id} fill:#c8e6c9,stroke:#388e3c,color:#333;",
        "end": "    style {id} fill:#ffcdd2,stroke:#d32f2f,color:#333;",
    }
    
    # Apply styles
    for node in graph_dict["nodes"]:
        node_type = node.get("type", "process")
        if node_type in style_defs:
            mermaid_code.append(style_defs[node_type].format(id=node["id"]))
    
    # Add edges
    for edge in graph_dict["edges"]:
        from_node = edge["from"]
        to_node = edge["to"]
        transition = edge.get("metadata", {}).get("transition", "")
        if transition:
            mermaid_code.append(f"    {from_node} -->|{transition}| {to_node};")
        else:
            mermaid_code.append(f"    {from_node} --> {to_node};")
    
    return "\n".join(mermaid_code)

def create_graph_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a visualization graph from the LangGraph state.
    
    Args:
        state: Research agent state after execution
        
    Returns:
        Dictionary with nodes and edges for visualization
    """
    # Define all possible nodes
    nodes = [
        {"id": "start", "label": "User Question", "type": "start"},
        {"id": "web_search", "label": "Web Search", "type": "process"},
        {"id": "academic_search", "label": "Academic Search", "type": "process"},
        {"id": "youtube_search", "label": "YouTube Search", "type": "process"},
        {"id": "social_media_search", "label": "Social Media Search", "type": "process"},
        {"id": "reflect", "label": "Self Reflection", "type": "process"},
        {"id": "summarize", "label": "Final Summary", "type": "end"},
    ]
    
    # Extract all used sources from logs
    used_sources = {"start"}  # Always include start node
    edges = []
    
    # Track the sequence of operations from search logs
    search_sources = []
    for log in state.get("search_logs", []):
        source = log["source"]
        search_sources.append(source)
        used_sources.add(f"{source}_search")
    
    # If we have any searches, add reflection and summary nodes
    if search_sources:
        used_sources.add("reflect")
        used_sources.add("summarize")
        
        # Add edges from start to all search nodes
        current_step = 1
        for source in set(search_sources):  # Use set to handle duplicates in loops
            edges.append({
                "from": "start",
                "to": f"{source}_search",
                "metadata": {"transition": current_step}
            })
        
        # Add edges from all search nodes to reflection
        current_step += 1
        for source in set(search_sources):
            edges.append({
                "from": f"{source}_search",
                "to": "reflect",
                "metadata": {"transition": current_step}
            })
        
        # Add edge from reflection to summary
        current_step += 1
        edges.append({
            "from": "reflect",
            "to": "summarize",
            "metadata": {"transition": current_step}
        })
    else:
        # If no searches were performed, connect start directly to summary
        used_sources.add("summarize")
        edges.append({
            "from": "start",
            "to": "summarize",
            "metadata": {"transition": 1}
        })
    
    # Filter nodes to only include those used in the workflow
    filtered_nodes = [node for node in nodes if node["id"] in used_sources]
    
    return {
        "nodes": filtered_nodes,
        "edges": edges
    }

def research_state_to_mermaid(state: Dict[str, Any]) -> str:
    """
    Convert research agent state to Mermaid diagram code.
    
    Args:
        state: Research agent state after execution
        
    Returns:
        Mermaid diagram code as a string
    """
    graph_dict = create_graph_from_state(state)
    return graph_to_mermaid(graph_dict) 