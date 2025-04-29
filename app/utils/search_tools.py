import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
import arxiv
import requests
from scholarly import scholarly
from pytube import Search as YouTubeSearch
import tweepy
from linkedin_api import Linkedin
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from duckduckgo_search import DDGS
from tavily import TavilyClient

from app.utils.config import config

# Web Search using SerpAPI
class CustomSerpAPIWrapper(SerpAPIWrapper):
    def __init__(self):
        super().__init__(serpapi_api_key=config.serpapi_api_key)
        
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Run query through SerpAPI and return formatted results."""
        search_results = super().results(query)
        
        if not search_results.get("organic_results"):
            return []
        
        formatted_results = []
        for result in search_results["organic_results"][:max_results]:
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "SerpAPI"
            })
        
        return formatted_results

# DuckDuckGo Search
def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("body", ""),
                    "source": "DuckDuckGo"
                })
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "title": result["title"],
                    "link": result["link"],  # Remove "DuckDuckGo Link:" prefix
                    "snippet": result["snippet"],
                    "source": "DuckDuckGo"
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        return []

# Tavily Search
def search_tavily(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using Tavily and return formatted results."""
    try:
        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(query=query, search_depth="basic", max_results=max_results)
        
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("url", ""),
                "snippet": result.get("content", ""),
                "source": "Tavily"
            })
        
        return formatted_results
    except Exception as e:
        print(f"Error searching Tavily: {str(e)}")
        return []

# arXiv Paper Search
def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search for papers on arXiv and return results."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for paper in search.results():
        # Use entry_id for HTML link instead of pdf_url
        paper_url = paper.entry_id if paper.entry_id else paper.pdf_url
        results.append({
            "title": paper.title,
            "authors": ", ".join(author.name for author in paper.authors),
            "summary": paper.summary,
            "url": paper_url,  # Use consistent url field
            "published": paper.published.strftime("%Y-%m-%d"),
            "source": "arxiv"
        })
    
    return results

# Google Scholar Search
def search_google_scholar(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search for papers on Google Scholar and return results."""
    search_query = scholarly.search_pubs(query)
    
    results = []
    for _ in range(max_results):
        try:
            paper = next(search_query)
            
            # Extract information safely with fallbacks
            title = paper.get('bib', {}).get('title', 'No Title')
            authors = ", ".join(paper.get('bib', {}).get('author', ['Unknown']))
            abstract = paper.get('bib', {}).get('abstract', 'No abstract available')
            
            # Try multiple URL sources
            url = paper.get('pub_url')  # Try official URL first
            if not url or url == 'No URL available':
                url = paper.get('url_scholarbib', '')  # Try Scholar URL
            if not url:
                url = f"https://scholar.google.com/scholar?cluster={paper.get('cluster_id', '')}"  # Fallback to Scholar search
            
            year = paper.get('bib', {}).get('pub_year', 'Unknown')
            
            results.append({
                "title": title,
                "authors": authors,
                "summary": abstract,
                "url": url,  # Use consistent url field
                "published": str(year),
                "source": "google_scholar"
            })
        except StopIteration:
            break
        except Exception as e:
            print(f"Error retrieving Google Scholar results: {str(e)}")
            continue
    
    return results

# YouTube Video Search
def search_youtube(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search for videos on YouTube and return results."""
    try:
        search = YouTubeSearch(query)
        results = []
        
        # Get the raw search results
        raw_results = search.results
        if not raw_results:
            return results
        
        # Process each video, with better error handling
        for video in raw_results[:max_results]:
            try:
                # Extract video ID from the video object
                video_id = None
                if hasattr(video, 'video_id'):
                    video_id = video.video_id
                elif hasattr(video, 'id'):
                    video_id = video.id
                elif hasattr(video, 'watch_url'):
                    video_id = video.watch_url.split('v=')[-1].split('&')[0]
                
                if not video_id:
                    continue
                
                # Build result with safe attribute access
                result = {
                    "title": getattr(video, 'title', 'No title available'),
                    "channel": getattr(video, 'author', 'Unknown channel'),
                    "description": getattr(video, 'description', 'No description available'),
                    "link": f"https://www.youtube.com/watch?v={video_id}",
                    "publish_date": str(getattr(video, 'publish_date', 'Date unknown')),
                    "source": "youtube"
                }
                
                # Only add if we have at least a title and link
                if result["title"] != 'No title available' and video_id:
                    results.append(result)
                    
            except (AttributeError, KeyError, IndexError) as e:
                print(f"Error processing YouTube video: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        print(f"Error in YouTube search: {str(e)}")
        return []

# Twitter Search
def search_twitter(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search for tweets on Twitter and return results."""
    # Set up Twitter API
    auth = tweepy.OAuth1UserHandler(
        config.twitter_api_key,
        config.twitter_api_secret,
        config.twitter_access_token,
        config.twitter_access_token_secret
    )
    api = tweepy.API(auth)
    
    results = []
    try:
        tweets = api.search_tweets(q=query, count=max_results, tweet_mode="extended")
        
        for tweet in tweets:
            results.append({
                "text": tweet.full_text,
                "user": tweet.user.screen_name,
                "created_at": str(tweet.created_at),
                "link": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                "source": "twitter"
            })
    except Exception as e:
        print(f"Error searching Twitter: {str(e)}")
    
    return results

# LinkedIn Search
def search_linkedin(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search for posts on LinkedIn and return results."""
    try:
        # Authenticate with LinkedIn
        api = Linkedin(config.linkedin_username, config.linkedin_password)
        
        # Search for posts
        search_results = api.search_people(query)
        
        results = []
        for profile in search_results[:max_results]:
            profile_id = profile.get('public_id', '')
            if profile_id:
                # Get recent posts from this profile
                try:
                    posts = api.get_profile_posts(profile_id, count=1)
                    if posts:
                        post = posts[0]
                        results.append({
                            "text": post.get('commentary', {}).get('text', 'No text available'),
                            "author": f"{profile.get('firstName', '')} {profile.get('lastName', '')}",
                            "link": f"https://www.linkedin.com/in/{profile_id}/",
                            "source": "linkedin"
                        })
                except Exception as e:
                    print(f"Error getting LinkedIn posts: {str(e)}")
                    continue
        
        return results
    except Exception as e:
        print(f"Error authenticating with LinkedIn: {str(e)}")
        return []

# Create LangChain tools
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    if not config.search_config.web_search_enabled:
        return "Web search is disabled."
    
    results = []
    errors = []
    
    # SerpAPI Search
    if config.search_config.web_search_config.use_serpapi:
        if config.is_api_configured("serpapi"):
            try:
                serpapi = CustomSerpAPIWrapper()
                serpapi_results = serpapi.search(query, max_results=config.max_search_results)
                results.extend(serpapi_results)
            except Exception as e:
                errors.append(f"Error with SerpAPI search: {str(e)}")
        else:
            errors.append("SerpAPI key is not configured.")
    
    # DuckDuckGo Search
    if config.search_config.web_search_config.use_duckduckgo:
        try:
            duckduckgo_results = search_duckduckgo(query, max_results=config.max_search_results)
            results.extend(duckduckgo_results)
        except Exception as e:
            errors.append(f"Error with DuckDuckGo search: {str(e)}")
    
    # Tavily Search
    if config.search_config.web_search_config.use_tavily:
        if config.is_api_configured("tavily"):
            try:
                tavily_results = search_tavily(query, max_results=config.max_search_results)
                results.extend(tavily_results)
            except Exception as e:
                errors.append(f"Error with Tavily search: {str(e)}")
        else:
            errors.append("Tavily API key is not configured.")
            
    # Firecrawl Search
    if config.search_config.web_search_config.use_firecrawl:
        if config.is_api_configured("firecrawl"):
            try:
                from mcp_firecrawl_firecrawl import deep_research
                # Set environment variable for Firecrawl
                os.environ["FIRECRAWL_API_KEY"] = config.firecrawl_api_key
                
                firecrawl_results = deep_research(
                    query=query,
                    maxUrls=config.max_search_results,
                    maxDepth=2,  # Conservative depth for regular search
                    timeLimit=60  # 1 minute timeout
                )
                
                if isinstance(firecrawl_results, dict):
                    # Extract sources from Firecrawl results
                    if "sources" in firecrawl_results:
                        for source in firecrawl_results["sources"]:
                            if isinstance(source, dict):
                                results.append({
                                    "title": source.get("title", "Untitled"),
                                    "link": source.get("url", ""),
                                    "snippet": source.get("snippet", ""),
                                    "source": "Firecrawl"
                                })
            except Exception as e:
                errors.append(f"Error with Firecrawl search: {str(e)}")
        else:
            errors.append("Firecrawl API key is not configured.")
    
    if not results:
        error_message = "\n".join(errors) if errors else "No web search results found."
        return error_message
    
    # Format results into markdown
    formatted_output = []
    for result in results:
        formatted_output.append(f"### {result['title']}")
        formatted_output.append(f"[{result['link']}]({result['link']})")
        formatted_output.append(f"\n{result['snippet']}\n")
    
    return "\n".join(formatted_output)

@tool
def academic_search(query: str) -> str:
    """Search for academic papers on the given query from arXiv and Google Scholar."""
    results = []
    formatted_results = ""
    
    # arXiv search
    if config.search_config.arxiv_search_enabled:
        try:
            arxiv_results = search_arxiv(query, max_results=config.max_arxiv_results)
            formatted_results += "### arXiv Papers\n\n"
            for paper in arxiv_results:
                formatted_results += f"Title: {paper['title'].strip()}\n"
                formatted_results += f"URL: {paper['url']}\n"
                formatted_results += f"Authors: {paper['authors']}\n"
                formatted_results += f"Published: {paper['published']}\n"
                formatted_results += f"Abstract: {paper['summary'].strip()}\n\n"
            results.extend(arxiv_results)
        except Exception as e:
            print(f"Error searching arXiv: {str(e)}")
            formatted_results += "Error searching arXiv papers.\n\n"
    
    # Google Scholar search
    if config.search_config.scholar_search_enabled:
        try:
            scholar_results = search_google_scholar(query, max_results=config.max_scholar_results)
            formatted_results += "### Google Scholar Papers\n\n"
            for paper in scholar_results:
                # Skip papers without essential information
                if not paper['title']:
                    continue
                    
                formatted_results += f"Title: {paper['title'].strip()}\n"
                formatted_results += f"URL: {paper['url']}\n"
                formatted_results += f"Authors: {paper['authors']}\n"
                formatted_results += f"Published: {paper['published']}\n"
                # Handle cases where abstract might not be available
                abstract = paper['summary'] if paper['summary'] != 'No abstract available' else 'Abstract not available'
                formatted_results += f"Abstract: {abstract.strip()}\n\n"
            results.extend(scholar_results)
        except Exception as e:
            print(f"Error searching Google Scholar: {str(e)}")
            formatted_results += "Error searching Google Scholar papers.\n\n"
    
    if not results:
        return "No academic search results found or academic search is disabled."
    
    return formatted_results

@tool
def youtube_search(query: str) -> str:
    """Search for YouTube videos on the given query."""
    if not config.search_config.youtube_search_enabled:
        return "YouTube search is disabled."
    
    try:
        results = search_youtube(query, max_results=config.max_youtube_results)
        
        if not results:
            return "No YouTube video results found."
        
        formatted_results = ""
        for result in results:
            formatted_results += f"Title: {result['title']}\n"
            formatted_results += f"URL: {result['link']}\n"
            formatted_results += f"Description: {result['description']}\n\n"
        
        return formatted_results
    except Exception as e:
        print(f"Error searching YouTube: {str(e)}")
        return f"Error searching YouTube: {str(e)}"

@tool
def social_media_search(query: str) -> str:
    """Search for related content on Twitter and LinkedIn."""
    results = []
    
    # Twitter search
    if config.search_config.twitter_search_enabled:
        if config.is_api_configured("twitter"):
            try:
                twitter_results = search_twitter(query, max_results=config.max_twitter_results)
                results.extend(twitter_results)
            except Exception as e:
                results.append({"error": f"Error searching Twitter: {str(e)}", "source": "twitter"})
        else:
            results.append({"error": "Twitter API is not configured.", "source": "twitter"})
    
    # LinkedIn search
    if config.search_config.linkedin_search_enabled:
        if config.is_api_configured("linkedin"):
            try:
                linkedin_results = search_linkedin(query, max_results=config.max_linkedin_results)
                results.extend(linkedin_results)
            except Exception as e:
                results.append({"error": f"Error searching LinkedIn: {str(e)}", "source": "linkedin"})
        else:
            results.append({"error": "LinkedIn API is not configured.", "source": "linkedin"})
    
    if not results:
        return "Social media search is disabled or no results found."
    
    formatted_results = "Social Media Results:\n\n"
    for i, result in enumerate(results, 1):
        if "error" in result:
            formatted_results += f"{result['error']}\n\n"
            continue
            
        if result.get("source") == "twitter":
            formatted_results += f"{i}. Tweet by @{result['user']}\n"
            formatted_results += f"   Posted on: {result['created_at']}\n"
            formatted_results += f"   Content: {result['text']}\n"
            formatted_results += f"   Link: {result['link']}\n\n"
        elif result.get("source") == "linkedin":
            formatted_results += f"{i}. LinkedIn Post by {result['author']}\n"
            formatted_results += f"   Content: {result['text']}\n"
            formatted_results += f"   Profile: {result['link']}\n\n"
    
    return formatted_results 