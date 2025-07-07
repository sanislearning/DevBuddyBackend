# devbuddy_app.py

import os
import traceback
import streamlit as st
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
import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# Step 1: Crawl documentation from user-given URL
async def crawl_docs(user_url):
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=LLMContentFilter(
            llm_config=LLMConfig(provider="gemini/gemini-2.0-flash", api_token=os.getenv("GOOGLE_API_KEY")),
            instruction="""
                Extract tutorials, guides, explanations, documentation, and code examples.
                Ignore navigation menus, repetitive headers/footers, and irrelevant boilerplate.
                Format the output in clean Markdown format.
            """,
            verbose=True
        )
    )

    config = CrawlerRunConfig(
        markdown_generator=markdown_generator,
        deep_crawl_strategy=BFSDeepCrawlStrategy(max_pages=5, max_depth=2),
        cache_mode="bypass"
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(user_url, config=config)

        # Combine markdown from all pages
        combined_markdown = ""
        for result in results:
            content = result.markdown.fit_markdown.strip()
            if content:
                combined_markdown += f"# URL: {result.url}\n"
                combined_markdown += f"**Depth**: {result.metadata.get('depth', 0)}\n\n"
                combined_markdown += content + "\n\n---\n\n"

        return combined_markdown


# Step 2: Split, embed and store markdown
def process_markdown(md_text):
    # Save markdown to a temporary file
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


# Step 3: Create the RAG chain
def create_rag_chain(vectordb):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    retriever = vectordb.as_retriever(search_type='similarity', k=10)

    system_prompt = (
        "You are DevBuddy, a helpful developer assistant. Use the provided documentation to answer questions about programming tools, libraries, and frameworks.\n"
        "When a user asks for a 'boilerplate', 'template', or 'example', respond with clean and working code that is typical for the use case.\n"
        "If the documentation includes example code, extract and adapt it. Explain the code well in a concise but complete.\n"
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


# Step 4: Streamlit UI
st.set_page_config(page_title="DevBuddy", layout="centered")
st.title("🤖 DevBuddy – Your AI Dev Documentation Assistant")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Input URL
user_url = st.text_input("🔗 Enter the documentation URL to crawl:")

if st.button("Start Crawling") and user_url:
    with st.spinner("🕸️ Crawling and extracting content..."):
        try:
            md_text = asyncio.run(crawl_docs(user_url))
            vectordb = process_markdown(md_text)
            rag_chain = create_rag_chain(vectordb)

            st.session_state.vectordb = vectordb
            st.session_state.rag_chain = rag_chain

            st.success("✅ Ready! Ask your questions below.")
        except Exception as e:
            st.error("❌ Error while crawling or processing.")
            traceback.print_exc()

# Chat input and response
if "rag_chain" in st.session_state:
    user_question = st.chat_input("Ask a question about the documentation...")

    if user_question:
        st.session_state.chat_memory.append({"role": "user", "content": user_question})
        history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_memory
        )

        with st.spinner("🤖 Thinking..."):
            response = st.session_state.rag_chain.invoke({"input": history})
            answer = response["answer"]

            st.session_state.chat_memory.append({"role": "assistant", "content": answer})

            st.chat_message("user").markdown(user_question)
            st.chat_message("assistant").markdown(answer)
