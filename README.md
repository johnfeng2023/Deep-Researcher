# 🔍 Deep Researcher

Deep Researcher is a powerful research agent built with LangChain and LangGraph that helps you conduct comprehensive research on any topic by leveraging multiple data sources.

![homepage](assets/homepage.png)

## 💡 Features

- Answer research queries using AI-powered reasoning
- Search the web using multiple search engines:
  - SerpAPI (Google search results)
  - DuckDuckGo (privacy-focused search)
  - Tavily (AI-powered search optimized for research)
- Search academic papers from arXiv and Google Scholar
- Find relevant YouTube videos
- Discover insights from Twitter and LinkedIn posts
- Upload local documents to include in research
- Send research results via email
- Interactive visualization of the agent's reasoning process
- Customizable search options - enable/disable specific search features

## 🔧 Installation

For detailed installation instructions and troubleshooting tips, please see the [Installation Guide](INSTALLATION.md).

Quick start:

1. Clone this repository:
```bash
git clone https://github.com/johnfeng2023/Deep-Researcher.git
cd Deep-Researcher
```

2. Create and activate a virtual environment:
```bash
python - m venv .venv
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Ollama locally for the LLM:
```bash
# Install Ollama if not already installed
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Gemma model
ollama pull gemma3:1b
```

5. Create a `.env` file in the root directory with your API keys (see `.env.example`).

## 🔭 Usage

1. Start the Streamlit app:
```bash
export PYTHONPATH=$(pwd)
streamlit run app/main.py
```
Or use the provided shell script:
```bash
./run.sh
```

2. Open your browser and go to `http://localhost:8501`

3. Enter your research query, select the search options you want to enable, and click "Research"

4. View the results, LangGraph visualization, and send the results to an email if desired

## 🌐 Web Search Options

Deep Researcher offers multiple web search options:

- **SerpAPI**: Provides Google search results (requires API key)
- **DuckDuckGo**: Privacy-focused search engine (no API key required)
- **Tavily**: AI-powered search engine optimized for research (requires API key)

You can enable or disable any of these search engines from the sidebar.

## 🧩 Architecture

This project uses:
- LangChain for orchestrating the various AI components and tools
- LangGraph for creating a dynamic research workflow with self-reflection capabilities
- Ollama (gemma3:1b) for local AI model inference
- Streamlit for the web interface

<!--- 
# Deep Researcher

An advanced research assistant system that uses artificial intelligence to conduct multi source research, integrating large language models with various search tools and databases for automated scholarly research.

## Overview

Deep Researcher offers significant capabilities in:
- Processing complex research queries
- Gathering information from diverse sources
- Synthesizing coherent research summaries
- Providing a simple, intuitive interface

## 1. Introduction and Scope

### 1.1 Problem Statement

Conducting research requires navigating vast amounts of information spread across many platforms, including academic databases, web search engines, video platforms, and social media. The fragmented nature of this information landscape poses significant challenges:
- Time intensive manual search processes across multiple platforms
- Difficulty in synthesizing information from heterogeneous sources
- Inefficient evaluation of source relevance and reliability
- Limited ability to process large volumes of information efficiently

### 1.2 Project Goals and Objectives

We developed Deep Researcher to address these challenges with the following objectives:
- Create an autonomous research agent capable of conducting comprehensive research across multiple information sources
- Implement a self reflective research process that can evaluate information gaps and dynamically adjust search strategies
- Develop an integrated search system spanning academic databases, web search engines, multimedia platforms, and social media
- Incorporate local document processing capabilities to include user inputted info in the research process
- Present research findings in a clear, coherent format with proper attribution and supporting evidence
- Provide an intuitive user interface allowing researchers to configure search parameters and visualize the research process

## 2. Methodological Approach

### 2.1 System Architecture Design

The Deep Researcher system was designed with a modular architecture centered around a state-based research workflow. The core components include:
- **Research Agent**: A LangGraph powered state machine that orchestrates the research process
- **Search Tools**: Modular components providing access to various information sources
- **Document Processing System**: Tools for extracting and processing information from local documents
- **RAG System**: A retrieval-augmented generation system for integrating local knowledge
- **User Interface**: A Streamlit based interface for configuration and visualization

### 2.2 Development Process

The development followed an iterative approach with these key phases:
- **Requirement Analysis**: Identifying essential information sources and research capabilities
- **Component Development**: Creating modular tools for each information source
- **Workflow Design**: Designing the research agent's state graph and decision processes
- **Integration**: Connecting components through the LangChain framework
- **User Interface Development**: Creating an intuitive Streamlit interface
- **Testing and Refinement**: Iterative testing with diverse research queries

### 2.3 Research Agent Implementation

The research agent was implemented as a directed state graph using LangGraph with the following key states:
- **Query Analysis**: Understanding the research question and formulating search strategies
- **Source Selection**: Dynamically selecting appropriate information sources
- **Information Gathering**: Executing searches across selected platforms
- **Reflection**: Evaluating gathered information to identify knowledge gaps
- **Synthesis**: Generating a comprehensive research summary

The agent employs self-reflection mechanisms to evaluate information quality, identify gaps, and decide when sufficient information has been gathered to answer the research question.

## 3. Technical Implementation

### 3.1 Core Technologies

The system leverages several key technologies:
- **LangChain**: Provides the foundation for connecting language models with external tools
- **LangGraph**: Enables the creation of a stateful agent with reasoning capabilities
- **Ollama/Gemma 3**: Provides local LLM inference capabilities
- **FAISS**: Powers the vector database for document retrieval
- **Streamlit**: Creates the web-based user interface

### 3.2 Search Integration

The system integrates multiple search capabilities:

**Web Search**:
- SerpAPI for Google search results
- DuckDuckGo for privacy-focused search
- Tavily for AI optimized research queries

**Academic Sources**:
- arXiv for scientific papers
- Google Scholar for academic publications

**Multimedia Sources**:
- YouTube for video content

**Social Media**:
- Twitter/X for current discussions
- LinkedIn for professional insights

Each search tool was implemented as a LangChain tool using appropriate APIs or web scraping techniques where necessary.

### 3.3 Document Processing System

The document processing system supports:
- Multiple file formats (PDF, DOCX, TXT, MD, CSV)
- Automatic text extraction
- Chunking for efficient embedding
- Vector-based similarity search

### 3.4 RAG Implementation

The RAG (Retrieval-Augmented Generation) system:
- Processes uploaded documents into text chunks
- Creates embeddings using the all-MiniLM-L6-v2 model
- Stores embeddings in a FAISS vector database
- Retrieves contextually relevant information during research
- Integrates retrieved information into the research process

### 3.5 Model Context Processor (MCP)
The Model Context Processor is a critical component that enhances the system's ability to handle and process information effectively:
- Dynamic Context Management: Intelligently manages the context window for large language models
- Document Prioritization: Ranks and prioritizes the most relevant information for inclusion in the context
- Context Optimization: Ensures the most valuable information is preserved when context limitations are reached
- Information Density Analysis: Evaluates content for information density to maximize utility within token limits
- Adaptive Chunking: Dynamically adjusts document chunking based on content complexity and relevance
- Memory Management: Maintains a hierarchical memory structure to preserve key insights across research stages

The MCP significantly improves research quality by ensuring the language model has access to the most relevant information at each stage of the research process, effectively overcoming context window limitations inherent to most LLMs.

## 4. Results and Findings

### 4.1 Research Capability Assessment

The Deep Researcher system demonstrated strong capabilities in:
- **Comprehensive Research**: Successfully gathering information from multiple sources for diverse queries
- **Information Synthesis**: Creating coherent summaries that integrate findings across sources
- **Self-Reflection**: Effectively identifying knowledge gaps and adjusting search strategies
- **Source Citation**: Properly attributing information to original sources

### 4.2 Performance Characteristics

Key performance characteristics include:
- **Response Time**: Typically 2-5 minutes for comprehensive research, depending on query complexity
- **Information Diversity**: Successfully integrates information from 3-7 distinct sources per query
- **Accuracy**: High relevance of retrieved information, with appropriate citations and source tracking

### 4.3 User Experience

The Streamlit interface provides:
- Intuitive query input
- Configurable search options
- Real-time visualization of the research process
- Clear presentation of results with source attribution
- Email functionality for sharing results

## 5. Challenges and Solutions

### 5.1 Technical Challenges

**API Rate Limiting**:
- Challenge: Search APIs often impose rate limits
- Solution: Implemented rate limiting, caching, and fallback mechanisms

**API Costs**:
- Challenge: Some social media API's require payment to access
- Solution: Implemented the infrastructure for such API's without subscribing to the actual platform

**Local LLM Performance**:
- Challenge: Local LLMs have lower performance than cloud-based alternatives
- Solution: Carefully engineered prompts and implemented a reflection mechanism to compensate for limitations

**RAG Quality Issues**:
- Challenge: Document chunking and retrieval sometimes missed contextual information
- Solution: Optimized chunk size and overlap parameters, implemented semantic search improvements

**Integration Complexity**:
- Challenge: Coordinating multiple search tools within a coherent workflow
- Solution: LangGraph-based state management with conditional transitions

### 5.2 Research Quality Challenges

**Information Synthesis**:
- Challenge: Creating coherent narratives from heterogeneous sources
- Solution: Implemented a multi-stage summarization process with contextual awareness

**Search Query Formulation**:
- Challenge: Generic search queries yielded poor results
- Solution: Added a query refinement step that tailors queries to each search platform

**Source Reliability**:
- Challenge: Evaluating the reliability of diverse information sources
- Solution: Implemented source credibility assessment within the reflection stage

## 6. Conclusion and Future Work

### 6.1 Achievements

The Deep Researcher project successfully created an autonomous research agent capable of:
- Conducting comprehensive research across multiple information sources
- Self-reflecting on research progress to identify knowledge gaps
- Synthesizing coherent research summaries with proper attribution
- Integrating user-provided documents into the research process

### 6.2 Limitations

Notable limitations include:
- Dependence on third-party APIs for certain search capabilities
- Computational requirements for running local LLMs
- Limited capabilities for evaluating source credibility
- Potential for search bias based on selected search engines

### 6.3 Future Work

Potential areas for enhancement include:
- **Improved Source Evaluation**: Implementing more sophisticated source credibility assessment
- **Expanded Academic Access**: Integration with additional academic databases
- **Multi-Modal Research**: Better processing of image and video content
- **Collaborative Research**: Adding capabilities for multiple users to contribute to research
- **Knowledge Base Building**: Creating persistent research memory across sessions

## References

- LangChain Documentation: https://python.langchain.com/docs/
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Ollama Project: https://github.com/ollama/ollama
- FAISS Vector Database: https://github.com/facebookresearch/faiss
- Streamlit Documentation: https://docs.streamlit.io/
-->

## 📺 Video Demonstrations

### Search Agent Demo
[![Search Agent Demo](https://img.youtube.com/vi/YklrWpmjqpU/0.jpg)](https://www.youtube.com/watch?v=YklrWpmjqpU)

### System Overview
<img width="569" alt="Screenshot 2025-05-01 at 5 10 31 PM" src="https://github.com/user-attachments/assets/f155b55a-d391-4f2a-b3a4-ea08901275b8" />

### RAG Implementation Demo
[![RAG Demo](https://img.youtube.com/vi/CvVQr5Tqkv8/0.jpg)](https://www.youtube.com/watch?v=CvVQr5Tqkv8)

### MCP Demo
[![MCP Demo](https://img.youtube.com/vi/cThGqxrcEa4/0.jpg)](https://www.youtube.com/watch?v=cThGqxrcEa4)

---
This software is for personal, non-commercial use only. Redistribution or modification is strictly prohibited.
