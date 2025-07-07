from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from dotenv import load_dotenv
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pureTerminal import crawl_docs,process_markdown,create_rag_chain
#You can import all of the functions that you defined in the scripts
#You can define how exactly your output are supposed to be via pydantic

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
    "http://127.0.0.1:5173"
    ], #change this to the react app url, don't forget
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Setting load env vars and set key
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#Required for Windows asyncio
if asyncio.get_event_loop().is_closed():
    asyncio.set_event_loop(asyncio.new_event_loop())

#RAG chain object
rag_chain=None

class CrawlRequest(BaseModel): #pydantic class describing the return format for the endpoint
    url:str

class AskRequest(BaseModel): #pydantic class describing the inputs to the model in from of questions
    question:str

@app.post("/crawl")
async def crawl_docs_api(req:CrawlRequest):
    global rag_chain
    markdown=await crawl_docs(req.url)
    vectordb=process_markdown(markdown)
    rag_chain=create_rag_chain(vectordb)
    return {"status":"Crawling and processing complete."}

@app.post("/ask")
async def ask_api(req:AskRequest):
    if not rag_chain:
        return {"answer":"Please crawl documentation first."}
    response=rag_chain.invoke({"input":req.question})
    return {"answer":response["answer"]}