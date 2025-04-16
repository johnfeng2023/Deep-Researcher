import re
import json
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.utils.config import config
from app.models.llm import (
    get_ollama_llm, 
    create_research_chain, 
    create_reflection_chain,
    create_summary_chain
)
from app.utils.search_tools import (
    web_search,
    academic_search,
    youtube_search,
    social_media_search
)

# Define the state for our research agent
class SearchLog(TypedDict):
    source: str
    query: str
    results: str
    links: Optional[List[Dict[str, str]]]  # Store URLs, titles, and abstracts

class ResearchState(TypedDict):
    question: str
    search_logs: List[SearchLog]
    current_answer: str
    final_answer: Optional[str]
    needs_more_research: bool
    next_search_strategy: Optional[str]
    background_info: Optional[str]
    key_areas: Optional[List[str]]
    challenges: Optional[List[str]]

# Define processing nodes for the graph
def route_search(state: ResearchState) -> Dict[str, Any]:
    """Route to the appropriate search based on the strategy."""
    strategy = state.get("next_search_strategy", "")
    
    if strategy == "web_search":
        return {"next": "web_search"}
    elif strategy == "academic_search":
        return {"next": "academic_search"}
    elif strategy == "youtube_search":
        return {"next": "youtube_search"}
    elif strategy == "social_media_search":
        return {"next": "social_media_search"}
    elif strategy == "summarize":
        return {"next": "summarize"}
    else:
        # Default to answering if no strategy specified
        return {"next": "answer"}

def process_web_search(state: ResearchState) -> ResearchState:
    """Execute web search and update the state."""
    search_query = state["question"]
    search_result = web_search.run(search_query)
    
    # Add to search logs
    search_logs = state.get("search_logs", [])
    search_logs.append({
        "source": "web",
        "query": search_query,
        "results": search_result,
        "links": []  # Web search results are already processed
    })
    
    # Update state
    current_answer = state.get("current_answer", "")
    current_answer += f"\n\n## Web Search Results\n{search_result}"
    
    return {
        **state,
        "search_logs": search_logs,
        "current_answer": current_answer,
        "next_search_strategy": None  # Clear strategy for reflection phase
    }

def process_academic_search(state: ResearchState) -> ResearchState:
    """Execute academic search and update the state."""
    search_query = state["question"]
    search_result = academic_search.run(search_query)
    
    # Initialize formatted result and papers list
    formatted_result = "## Academic Sources\n\n"
    papers = []
    
    # Process the search results section by section
    sections = search_result.split("\n\n")
    current_paper = {}
    
    for section in sections:
        if not section.strip():
            continue
            
        # Start a new paper when we see a title
        if section.startswith("Title:"):
            # Save the previous paper if it exists and has required fields
            if current_paper and all(k in current_paper for k in ['title', 'url', 'abstract']):
                papers.append(current_paper.copy())
            current_paper = {}
            
            # Process each line in the section
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("Title:"):
                    current_paper['title'] = line[6:].strip()
                elif line.startswith("URL:"):
                    current_paper['url'] = line[4:].strip()
                elif line.startswith("Authors:"):
                    current_paper['authors'] = line[8:].strip()
                elif line.startswith("Published:"):
                    current_paper['published'] = line[10:].strip()
                elif line.startswith("Abstract:"):
                    current_paper['abstract'] = line[9:].strip()
    
    # Add the last paper if it exists and has required fields
    if current_paper and all(k in current_paper for k in ['title', 'url', 'abstract']):
        papers.append(current_paper)
    
    # Format the papers into markdown with better styling
    if papers:
        for i, paper in enumerate(papers, 1):
            formatted_result += f"### {i}. [{paper['title']}]({paper['url']})\n\n"
            if 'authors' in paper:
                formatted_result += f"**Authors:** {paper['authors']}\n\n"
            if 'published' in paper:
                formatted_result += f"**Published:** {paper['published']}\n\n"
            if 'abstract' in paper:
                # Truncate long abstracts for better readability
                abstract = paper['abstract']
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                formatted_result += f"**Abstract:**\n{abstract}\n\n"
            formatted_result += "---\n\n"
    else:
        formatted_result += "*No academic papers found for this query.*\n\n"
    
    # Add to search logs with better structure
    search_logs = state.get("search_logs", [])
    search_logs.append({
        "source": "academic",
        "query": search_query,
        "results": formatted_result,
        "links": papers  # Store structured paper data for potential future use
    })
    
    # Update state with formatted content
    current_answer = state.get("current_answer", "")
    current_answer += f"\n\n{formatted_result}"
    
    return {
        **state,
        "search_logs": search_logs,
        "current_answer": current_answer,
        "next_search_strategy": None
    }

def process_youtube_search(state: ResearchState) -> ResearchState:
    """Execute YouTube search and update the state."""
    search_query = state["question"]
    search_result = youtube_search.run(search_query)
    
    # Extract video information
    videos = []
    current_video = {}
    
    # Process the search results
    for line in search_result.split('\n'):
        line = line.strip()
        if not line:
            if current_video and all(k in current_video for k in ['title', 'url']):
                videos.append(current_video.copy())
            current_video = {}
        elif line.startswith('Title:'):
            current_video['title'] = line[6:].strip()
        elif line.startswith('URL:'):
            current_video['url'] = line[4:].strip()
        elif line.startswith('Description:'):
            current_video['description'] = line[12:].strip()
    
    # Add the last video if it exists
    if current_video and all(k in current_video for k in ['title', 'url']):
        videos.append(current_video)
    
    # Format YouTube results with embedded videos
    formatted_result = "## Related Videos\n\n"
    if videos:
        for video in videos:
            formatted_result += f"### {video['title']}\n\n"
            # Extract video ID and create embedded player
            video_id = video['url'].split('v=')[-1].split('&')[0]  # Handle URLs with additional parameters
            formatted_result += f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>\n\n'
            if video.get('description'):
                formatted_result += f"{video['description']}\n\n"
            formatted_result += "---\n\n"
    else:
        formatted_result += "*No relevant videos found for this query.*\n\n"
    
    # Add to search logs
    search_logs = state.get("search_logs", [])
    search_logs.append({
        "source": "youtube",
        "query": search_query,
        "results": formatted_result,
        "links": videos
    })
    
    # Update state
    current_answer = state.get("current_answer", "")
    current_answer += f"\n\n{formatted_result}"
    
    return {
        **state,
        "search_logs": search_logs,
        "current_answer": current_answer,
        "next_search_strategy": None
    }

def process_social_media_search(state: ResearchState) -> ResearchState:
    """Execute social media search and update the state."""
    search_query = state["question"]
    search_result = social_media_search.run(search_query)
    
    # Add to search logs
    search_logs = state.get("search_logs", [])
    search_logs.append({
        "source": "social_media",
        "query": search_query,
        "results": search_result
    })
    
    # Update state
    current_answer = state.get("current_answer", "")
    current_answer += f"\n\n{search_result}"
    
    return {
        **state,
        "search_logs": search_logs,
        "current_answer": current_answer,
        "next_search_strategy": None  # Clear strategy for reflection phase
    }

def reflect_on_research(state: ResearchState) -> ResearchState:
    """Reflect on current findings and determine next search strategy."""
    reflection_chain = create_reflection_chain()
    
    reflection = reflection_chain.invoke({
        "question": state["question"],
        "findings": state["current_answer"]
    })
    
    # Extract if further research is needed from the natural text
    needs_more_research = "FURTHER_RESEARCH_NEEDED: Yes" in reflection
    
    # Get the number of loops completed
    loops_completed = len([log for log in state.get("search_logs", []) if log["source"] == "web"]) - 1
    
    # If we've already done 2 loops, force completion
    if loops_completed >= 2:
        needs_more_research = False
        next_search_strategy = "summarize"
    else:
        # Sequential search strategy
        next_search_strategy = None
        if needs_more_research:
            # Get all completed sources in this loop
            sources_used = [log["source"] for log in state.get("search_logs", [])]
            
            # Define the search sequence
            search_sequence = [
                ("web", config.search_config.web_search_enabled),
                ("academic", config.search_config.arxiv_search_enabled or config.search_config.scholar_search_enabled),
                ("youtube", config.search_config.youtube_search_enabled),
                ("social_media", config.search_config.twitter_search_enabled or config.search_config.linkedin_search_enabled)
            ]
            
            # Find the next search in the sequence
            for source, enabled in search_sequence:
                if source not in sources_used and enabled:
                    next_search_strategy = f"{source}_search"
                    break
            
            # If all sources have been used in this loop, start a new loop with web search
            if not next_search_strategy and loops_completed < 2:
                next_search_strategy = "web_search"
            elif not next_search_strategy:
                next_search_strategy = "summarize"
        else:
            # No more research needed, go to summarization
            next_search_strategy = "summarize"
    
    # Update state with reflection
    current_answer = state.get("current_answer", "")
    current_answer += f"\n\n### Research Progress Analysis\n{reflection}\n"
    
    return {
        **state,
        "current_answer": current_answer,
        "needs_more_research": needs_more_research,
        "next_search_strategy": next_search_strategy
    }

def create_final_summary(state: ResearchState) -> ResearchState:
    """Create a final summary of all the research findings."""
    summary_chain = create_summary_chain()
    
    # Organize the content by sections
    organized_content = f"# Research Analysis: {state['question']}\n\n"
    
    # Add all search results and reflections
    for log in state["search_logs"]:
        organized_content += log["results"] + "\n\n"
    
    # Generate the final analysis
    final_answer = summary_chain.invoke({
        "question": state["question"],
        "findings": organized_content
    })
    
    return {
        **state,
        "final_answer": final_answer,
        "needs_more_research": False,
    }

def should_continue(state: ResearchState) -> str:
    """Determine if research should continue or end."""
    if state.get("next_search_strategy") == "summarize":
        return "summarize"
    elif state.get("needs_more_research", True):
        return "continue"
    else:
        return "end"

def create_research_agent(
    search_config: Optional[Dict[str, bool]] = None
) -> StateGraph:
    """Create and configure the research agent workflow."""
    
    # Update search configuration if provided
    if search_config:
        config.update_search_config(search_config)
    
    # Create the workflow graph
    workflow = StateGraph(ResearchState)
    
    # Add our nodes
    workflow.add_node("route_search", route_search)
    workflow.add_node("web_search", process_web_search)
    workflow.add_node("academic_search", process_academic_search)
    workflow.add_node("youtube_search", process_youtube_search)
    workflow.add_node("social_media_search", process_social_media_search)
    workflow.add_node("reflect", reflect_on_research)
    workflow.add_node("summarize", create_final_summary)
    
    # Define edges
    # Initial routing based on the "next" key in the returned dictionary
    workflow.add_conditional_edges(
        "route_search",
        lambda x: x["next"],
        {
            "web_search": "web_search",
            "academic_search": "academic_search",
            "youtube_search": "youtube_search",
            "social_media_search": "social_media_search",
            "summarize": "summarize",
            "answer": "summarize",
        }
    )
    
    # Search results go to reflection
    workflow.add_edge("web_search", "reflect")
    workflow.add_edge("academic_search", "reflect")
    workflow.add_edge("youtube_search", "reflect")
    workflow.add_edge("social_media_search", "reflect")
    
    # After reflection, decide whether to continue or end
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "continue": "route_search",
            "summarize": "summarize",
            "end": END
        }
    )
    
    # After summarizing, end the workflow
    workflow.add_edge("summarize", END)
    
    # Set the entry point
    workflow.set_entry_point("route_search")
    
    return workflow

def run_research_agent(
    question: str,
    search_config: Optional[Dict[str, bool]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the research agent to answer a question.
    
    Args:
        question: The research question to answer
        search_config: Optional configuration for search tools
        
    Returns:
        Tuple of (final_answer, state)
    """
    # Create the research agent
    workflow = create_research_agent(search_config)
    
    # Compile the workflow into a runnable
    agent = workflow.compile()
    
    # Set initial state
    initial_state: ResearchState = {
        "question": question,
        "search_logs": [],
        "current_answer": "",
        "final_answer": None,
        "needs_more_research": True,
        "next_search_strategy": "web_search",  # Start with web search by default
        "background_info": None,
        "key_areas": None,
        "challenges": None
    }
    
    try:
        # Execute the workflow using the correct method
        result = agent.invoke(initial_state)
        
        # Extract the final answer
        final_answer = result.get("final_answer", "No answer was generated.")
        
        return final_answer, result
    except Exception as e:
        print(f"Error executing research agent: {str(e)}")
        return "An error occurred while processing your request.", {"error": str(e)} 