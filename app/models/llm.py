from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from typing import Any, Dict, List, Optional

from app.utils.config import config

def get_ollama_llm(temperature: float = 0.7, model: Optional[str] = None) -> Ollama:
    """Get a configured Ollama LLM instance."""
    model_name = model or config.ollama_model
    
    return Ollama(
        model=model_name,
        base_url=config.ollama_base_url,
        temperature=temperature,
    )

def create_research_chain(system_prompt: str) -> Any:
    """Create a research chain with the Ollama LLM."""
    llm = get_ollama_llm()
    
    prompt = PromptTemplate.from_template(
        """You are a helpful and friendly AI assistant. Please respond naturally and conversationally to general queries while maintaining professionalism.

1. Structure:
   - Begin with a clear, concise summary of the main points
   - Organize information in a logical, hierarchical manner
   - Use appropriate academic headings when necessary

2. Content Quality:
   - Prioritize peer-reviewed and scholarly sources
   - Distinguish between primary and secondary sources
   - Indicate levels of certainty in claims
   - Highlight any potential limitations or biases

3. Citation and References:
   - Properly cite sources when making specific claims
   - Include relevant quotes with proper attribution
   - Link to primary sources when available

4. Analysis:
   - Provide critical analysis rather than just summaries
   - Compare and contrast different viewpoints
   - Identify methodological strengths and weaknesses
   - Suggest areas for further research

5. Language:
   - Use precise, academic language
   - Avoid colloquialisms and informal expressions
   - Maintain objective, scholarly tone
   - Define technical terms when first used

User Question: {question}

Context: {context}

Please provide a comprehensive response following the guidelines above."""
    )
    
    chain = prompt | get_ollama_llm() | StrOutputParser()
    return chain

def create_reflection_chain():
    """Create a chain for reflecting on research progress."""
    template = """You are analyzing the current research findings to organize information and determine if more research is needed.

Question: {question}

Current Findings:
{findings}

Please analyze the findings and determine:
1. What are the key themes and insights from the research so far?
2. Are there any gaps in the current findings?
3. Are there conflicting viewpoints that need resolution?
4. What additional sources or perspectives would strengthen the research?

Format your response naturally but include somewhere in your text the following marker:
FURTHER_RESEARCH_NEEDED: <Yes/No>
"""

    prompt = PromptTemplate(
        input_variables=["question", "findings"],
        template=template
    )

    chain = prompt | get_ollama_llm() | StrOutputParser()
    return chain

def create_summary_chain():
    """Create a chain for generating the final summary."""
    template = """You are creating a comprehensive analysis of the research findings on {question}.

Research Findings:
{findings}

Please provide a scholarly analysis that:
1. Synthesizes the key findings and themes
2. Evaluates the quality and reliability of sources
3. Identifies areas of consensus and disagreement
4. Discusses methodological strengths and limitations
5. Suggests directions for future research

Remember to:
- Use precise academic language
- Properly cite sources
- Maintain a scholarly tone
- Define technical terms
- Indicate levels of certainty in claims

Structure your response in a clear, logical manner with appropriate headings and sections."""

    prompt = PromptTemplate(
        input_variables=["question", "findings"],
        template=template
    )

    chain = prompt | get_ollama_llm() | StrOutputParser()
    return chain 