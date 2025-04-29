import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class WebSearchConfig(BaseModel):
    use_serpapi: bool = Field(default=True, description="Use SerpAPI for web search")
    use_duckduckgo: bool = Field(default=True, description="Use DuckDuckGo for web search")
    use_tavily: bool = Field(default=False, description="Use Tavily for web search")
    use_firecrawl: bool = Field(default=False, description="Use Firecrawl for deep web search")

class SearchConfig(BaseModel):
    web_search_enabled: bool = Field(default=True, description="Enable web search using SerpAPI")
    arxiv_search_enabled: bool = Field(default=True, description="Enable arXiv paper search")
    scholar_search_enabled: bool = Field(default=True, description="Enable Google Scholar search")
    youtube_search_enabled: bool = Field(default=True, description="Enable YouTube video search")
    twitter_search_enabled: bool = Field(default=True, description="Enable Twitter search")
    linkedin_search_enabled: bool = Field(default=True, description="Enable LinkedIn search")
    
    # Web search engine selection
    web_search_config: WebSearchConfig = WebSearchConfig()

class RAGConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable RAG functionality")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="HuggingFace embedding model to use")
    chunk_size: int = Field(default=1000, description="Size of document chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between document chunks")
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")

class EmailConfig(BaseModel):
    username: str = Field(default_factory=lambda: os.getenv("EMAIL_USERNAME", ""))
    password: str = Field(default_factory=lambda: os.getenv("EMAIL_PASSWORD", ""))
    smtp_server: str = Field(default_factory=lambda: os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"))
    smtp_port: int = Field(default_factory=lambda: int(os.getenv("EMAIL_SMTP_PORT", "587")))

class Config(BaseModel):
    # API Keys
    serpapi_api_key: str = Field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    twitter_api_key: str = Field(default_factory=lambda: os.getenv("TWITTER_API_KEY", ""))
    twitter_api_secret: str = Field(default_factory=lambda: os.getenv("TWITTER_API_SECRET", ""))
    twitter_access_token: str = Field(default_factory=lambda: os.getenv("TWITTER_ACCESS_TOKEN", ""))
    twitter_access_token_secret: str = Field(default_factory=lambda: os.getenv("TWITTER_ACCESS_TOKEN_SECRET", ""))
    linkedin_username: str = Field(default_factory=lambda: os.getenv("LINKEDIN_USERNAME", ""))
    linkedin_password: str = Field(default_factory=lambda: os.getenv("LINKEDIN_PASSWORD", ""))
    firecrawl_api_key: str = Field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY", ""), description="Firecrawl API key")
    
    # LLM Config
    ollama_model: str = "gemma3:1b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Search Configuration
    search_config: SearchConfig = SearchConfig()
    
    # RAG Configuration
    rag_config: RAGConfig = RAGConfig()
    
    # Email Configuration
    email_config: EmailConfig = EmailConfig()
    
    # Customizable parameters
    max_search_results: int = 5
    max_arxiv_results: int = 5
    max_scholar_results: int = 5
    max_youtube_results: int = 3
    max_twitter_results: int = 5
    max_linkedin_results: int = 5
    
    def is_api_configured(self, api_name: str) -> bool:
        """Check if a particular API is configured with valid credentials"""
        if api_name == "serpapi":
            return bool(self.serpapi_api_key)
        elif api_name == "tavily":
            return bool(self.tavily_api_key)
        elif api_name == "twitter":
            return all([
                self.twitter_api_key,
                self.twitter_api_secret,
                self.twitter_access_token,
                self.twitter_access_token_secret
            ])
        elif api_name == "linkedin":
            return bool(self.linkedin_username and self.linkedin_password)
        elif api_name == "firecrawl":
            return bool(self.firecrawl_api_key)
        elif api_name == "email":
            return all([
                self.email_config.username,
                self.email_config.password,
                self.email_config.smtp_server,
                self.email_config.smtp_port
            ])
        return False
    
    def update_search_config(self, new_config: Dict[str, bool]) -> None:
        """Update search configuration with new settings"""
        for key, value in new_config.items():
            if key.startswith("web_search_config_"):
                # Handle web search engine selection
                engine_key = key.replace("web_search_config_", "")
                if hasattr(self.search_config.web_search_config, engine_key):
                    setattr(self.search_config.web_search_config, engine_key, value)
            elif key.startswith("rag_config_"):
                # Handle RAG configuration
                rag_key = key.replace("rag_config_", "")
                if hasattr(self.rag_config, rag_key):
                    setattr(self.rag_config, rag_key, value)
            elif hasattr(self.search_config, key):
                # Handle regular search options
                setattr(self.search_config, key, value)

# Create a global config instance
config = Config() 