from huggingface_hub import login

login("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")

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import json

from smolagents import Tool, CodeAgent, HfApiModel

##############################################################################
# 1) Define a web-scraping Tool
##############################################################################
from smolagents import Tool

class WebScrapeTool(Tool):
    """
    A tool that queries Google News RSS for 'digital health AI' references
    and returns a string containing JSON (with date/title/link, or an error).
    """

    name = "webscraper"
    description = (
        "Scrapes Google News RSS for references to 'digital health AI' stories. "
        "Returns a JSON string with 'error' and 'articles' fields."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The search query for Google News, e.g. 'digital health'. "
                "We'll build a Google News RSS URL from this query."
            )
        }
    }
    output_type = "string"  # Return a JSON string

    def forward(self, query: str):
        import json
        import requests
        from bs4 import BeautifulSoup

        # Build the Google News RSS URL based on the query
        search_url = f"https://news.google.com/rss/search?q={query}"

        try:
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            return json.dumps({"error": str(e), "articles": []})

        soup = BeautifulSoup(response.content, "xml")

        articles_found = []
        for item in soup.find_all("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            date_el = item.find("pubDate")

            title = title_el.text if title_el else "No Title"
            link = link_el.text if link_el else ""
            pub_date = date_el.text if date_el else ""

            articles_found.append({
                "title": title,
                "link": link,
                "date": pub_date
            })

        # Return a JSON string with error=None plus the articles array
        return json.dumps({"error": None, "articles": articles_found})

##############################################################################
# 2) Define a plotting Tool
##############################################################################
class PlotTrendTool(Tool):
    """
    Takes a JSON string of articles (date, title), plots them,
    and returns a success/failure message as a string.
    """

    name = "plot_trend"
    description = (
        "Plots the number of articles mentioning 'digital health AI' per day over time."
    )
    inputs = {
        "articles": {
            "type": "string",
            "description": "A JSON string with 'error' and 'articles' keys. 'articles' is a list of dicts with 'date'/'title'."
        }
    }
    output_type = "string"  # We'll just return a message string

    def forward(self, articles: str):
        # Parse the JSON string
        try:
            data = json.loads(articles)
        except Exception as e:
            return f"Failed to parse JSON: {str(e)}"

        if "error" in data and data["error"]:
            return f"Error returned by webscraper: {data['error']}"

        article_list = data.get("articles", [])
        if not article_list:
            return "No articles found or returned from the scraper."

        # Group counts by date
        counts_by_date = defaultdict(int)
        for art in article_list:
            d_str = art["date"]
            counts_by_date[d_str] += 1

        sorted_dates = sorted(counts_by_date.keys())
        if not sorted_dates:
            return "No articles to plot."

        x = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates]
        y = [counts_by_date[d] for d in sorted_dates]

        # Plot
        plt.figure(figsize=(8,4))
        plt.plot(x, y, marker='o')
        plt.title("Digital Health AI Articles Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("digital_health_ai_trend.png")
        plt.close()

        return "Plot saved to digital_health_ai_trend.png"

##############################################################################
# 3) Create and run a CodeAgent with these tools
##############################################################################
if __name__ == "__main__":
    # Initialize the tools
    webscraper_tool = WebScrapeTool()
    plot_tool = PlotTrendTool()

    # Example usage of HfApiModel
    agent = CodeAgent(
        tools=[webscraper_tool, plot_tool],
        additional_authorized_imports=["json", "bs4", "requests"],
        model=HfApiModel(
        "Qwen/Qwen2.5-Coder-32B-Instruct/v1",
        ),
        max_steps=15,
        verbosity_level=2,
    )

    # Natural language instruction
    user_query = (
        "Using beautifulsoup search known healthcare websites for digital health stories, then make a plot of how much they are trending. "
    )
    response = agent.run(user_query)
    print("Final Agent Output:\n", response)
