import os
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.utils.config import config
from app.models.llm import get_ollama_llm, create_research_chain
from app.utils.rag_utils import RAGSystem

# Define the state for our RAG agent
class RAGState(TypedDict):
    question: str
    context: List[str]
    retrieved_docs: List[Dict[str, Any]]
    rag_answer: Optional[str]

def create_rag_agent(collection_name: str = "default") -> StateGraph:
    """
    Create a RAG agent that retrieves relevant documents and generates an answer.
    
    Args:
        collection_name: Name of the vector collection to use
        
    Returns:
        A compiled graph that can be executed
    """
    # Initialize the RAG system
    rag_system = RAGSystem(collection_name=collection_name)
    
    # Define the workflow
    workflow = StateGraph(RAGState)
    
    # Define the document retrieval node
    def retrieve_documents(state: RAGState) -> RAGState:
        """Retrieve relevant documents for the question."""
        question = state["question"]
        
        # Retrieve documents
        retrieved_docs = rag_system.retrieve(question, k=config.max_search_results)
        
        # Format documents for the state
        formatted_docs = []
        context = []
        
        for doc in retrieved_docs:
            # Add formatted doc
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
            # Add to context
            context.append(doc.page_content)
        
        return {
            **state,
            "context": context,
            "retrieved_docs": formatted_docs
        }
    
    # Define the answer generation node
    def generate_answer(state: RAGState) -> RAGState:
        """Generate an answer based on the retrieved documents."""
        question = state["question"]
        context = "\n\n".join(state["context"])
        
        # Create the research chain
        system_prompt = """You are a helpful research assistant that provides accurate, informative answers based on the retrieved context.
        
Your task is to:
1. Synthesize information from the provided context
2. Structure your response with clear sections and headings if needed
3. Focus on addressing the specific question asked
4. Acknowledge limitations if the context doesn't provide sufficient information

Always maintain academic integrity in your responses."""
        
        chain = create_research_chain(system_prompt)
        
        # Generate the answer
        answer = chain.invoke({
            "question": question,
            "context": context
        })
        
        return {
            **state,
            "rag_answer": answer
        }
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("answer", generate_answer)
    
    # Add edges
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Compile the graph
    return workflow.compile()

def run_rag_agent(
    question: str,
    collection_name: str = "default"
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the RAG-enhanced research agent to answer a question.
    
    Args:
        question: The research question to answer
        collection_name: Name of the vector collection to use
        
    Returns:
        Tuple of (final_answer, state)
    """
    try:
        # Initialize RAG system only when this function is called
        rag_system = RAGSystem(collection_name=collection_name)
        
        # Get relevant documents
        docs = rag_system.get_relevant_documents(question)
        
        # Create initial state
        initial_state = {
            "question": question,
            "search_logs": [],
            "current_answer": "",
            "final_answer": None,
            "needs_more_research": True,
            "next_search_strategy": "web_search",
            "retrieved_docs": docs
        }
        
        # Run regular research agent with RAG context
        final_answer, state = run_research_agent(question=question)
        
        # Add retrieved documents to the state
        state["retrieved_docs"] = docs
        
        return final_answer, state
        
    except Exception as e:
        print(f"Error in RAG agent: {str(e)}")
        return "An error occurred while processing your request with RAG.", {"error": str(e)} 