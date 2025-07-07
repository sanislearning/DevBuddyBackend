#This works, current best option, now I just need to handle token utilisation a bit

import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai import LLMConfig
import os
from dotenv import load_dotenv

load_dotenv()
async def main():
    # Reintroduce LLMContentFilter with a more specific instruction
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=LLMContentFilter(
            llm_config=LLMConfig(provider="gemini/gemini-2.0-flash", api_token=os.getenv("GEMINI_API_KEY")),
            instruction="""
                        Extract the main content of the page: tutorials, explanations, guides, documentation, and relevant code examples. Ignore repetitive headers, footers, ads, and navigation menus. 
                        Do not exclude sections unless they are clearly irrelevant.
                        Format as clean Markdown.
                        """
                        ,
            verbose=True
        )
    )

    config = CrawlerRunConfig(
        markdown_generator=markdown_generator,
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_pages=10,
            max_depth=2,
            include_external=False
        ),
        cache_mode="bypass"
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun("https://python.langchain.com/docs/tutorials/agents/?utm_source=chatgpt.com", config=config)

        print(f"Crawled {len(results)} pages in total")

        # Access individual results
        with open("langchain_docs.md", "w", encoding="utf-8") as f:
            for result in results:
                url = result.url
                depth = result.metadata.get("depth", 0)
                markdown_content = result.markdown.fit_markdown.strip()

                if markdown_content:
                    f.write(f"# URL: {url}\n")
                    f.write(f"**Depth**: {depth}\n\n")
                    f.write(markdown_content)
                    f.write("\n\n---\n\n")
                else:
                    print(f"⚠️ No content found or content was filtered out by LLM: {url}")


if __name__ == "__main__":
    asyncio.run(main())