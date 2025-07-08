import os
import asyncio
import sys
import traceback
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai import LLMConfig

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def crawl_docs(user_url):
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=LLMContentFilter(
            llm_config=LLMConfig(provider="gemini/gemini-2.0-flash", api_token=os.getenv("GOOGLE_API_KEY")),
            instruction="""
                Extract tutorials, guides, explanations, documentation, boilerplate and code examples.
                Ignore navigation menus and repetitive headers/footers.
                Format the output in clean Markdown format.
            """,
            verbose=True
        )
    )

    config = CrawlerRunConfig(
        markdown_generator=markdown_generator,
        deep_crawl_strategy=BFSDeepCrawlStrategy(max_pages=20, max_depth=2),
        cache_mode="bypass"
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(user_url, config=config)

        combined_markdown = ""
        for result in results:
            content = result.markdown.fit_markdown.strip()
            if content:
                combined_markdown += f"# URL: {result.url}\n"
                combined_markdown += f"**Depth**: {result.metadata.get('depth', 0)}\n\n"
                combined_markdown += content + "\n\n---\n\n"

        return combined_markdown


def process_markdown(md_text):
    with open("temp_docs.md", "w", encoding="utf-8") as f:
        f.write(md_text)

    loader = TextLoader("temp_docs.md", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectordb = FAISS.from_documents(chunks, embedding)
    return vectordb


def create_rag_chain(vectordb):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    retriever = vectordb.as_retriever(search_type='similarity', k=10)

    system_prompt = (
        "You are DevBuddy, a helpful developer assistant. Use the provided documentation to answer questions about programming tools, libraries, and frameworks.\n"
        "When a user asks for a 'boilerplate', 'template', or 'example', respond with clean and working code that is typical for the use case.\n"
        "If the documentation includes example code, extract and adapt it. Be concise but complete.\n"
        "Avoid asking the user to refer to the docs themselves."
        "Avoid saying you cannot do something if the answer is present in the context.\n"
        "If unsure, say so clearly instead of hallucinating.\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


# === Main CLI Runner ===

def main():
    url = input("🔗 Enter the documentation URL to crawl: ").strip()

    if not url:
        print("❌ No URL provided. Exiting.")
        return

    print("🕸️ Crawling and extracting content...")
    try:
        md_text = asyncio.run(crawl_docs(url))
        vectordb = process_markdown(md_text)
        rag_chain = create_rag_chain(vectordb)
    except Exception as e:
        print("❌ Error during crawl or processing.")
        traceback.print_exc()
        return

    print("✅ DevBuddy is ready! Type your questions below (type `exit` to quit):\n")

    chat_history = [] #This one variable stores all of conversation

    while True:
        question = input("🧠 You: ")
        if question.lower().strip() == "exit":
            print("👋 Goodbye!")
            break

        chat_history.append({"role": "user", "content": question})
        history = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history)

        try:
            response = rag_chain.invoke({"input": history})
            answer = response["answer"]
        except Exception as e:
            answer = "⚠️ Something went wrong while generating the answer."
            traceback.print_exc()

        chat_history.append({"role": "assistant", "content": answer}) #Appends the chat history
        print(f"🤖 DevBuddy: {answer}\n") #prints out the answer


if __name__ == "__main__":
    main()
