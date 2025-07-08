from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from dotenv import load_dotenv
import sys
from fastapi.responses import JSONResponse
from fastapi import Request

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


@app.post("/crawl")
async def crawl_docs_api(req:CrawlRequest):
    global rag_chain
    markdown=await crawl_docs(req.url)
    vectordb=process_markdown(markdown)
    rag_chain=create_rag_chain(vectordb)
    return {"status":"Crawling and processing complete."}

@app.post("/api/chat")
async def chat_endpoint(request:Request):
    global rag_chain
    if not rag_chain:
        return JSONResponse(content={"error":"Please crawl documentation first."},status_code=400)
    try:
        data=await request.json()
        messages=data.get("messages",[])

        #Get latest user message
        user_message=next((m["content"] for m in reversed(messages) if m["role"]=="user"),None)

        if not user_message:
            return JSONResponse(content={"error":"No user message found."},status_code=400)
        print(f"🧠 Received messages: {messages}")
        print(f"💬 Latest user message: {user_message}")
        response=rag_chain.invoke({"input":user_message})
        print(response)
        return {"role":"assistant","content":response["answer"]}

    except Exception as e:
        return JSONResponse(content={"error":str(e)},status_code=500)