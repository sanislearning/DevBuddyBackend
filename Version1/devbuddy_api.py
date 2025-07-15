from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from dotenv import load_dotenv
import sys
from fastapi.responses import JSONResponse
from fastapi import Request
import traceback  

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pureTerminal import crawl_docs, process_markdown, create_rag_chain, summarize_history, MAX_HISTORY_LENGTH
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

chat_memory = []  # Optional: could be per user

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    global rag_chain
    if not rag_chain:
        return JSONResponse(content={"error": "Please crawl documentation first."}, status_code=400)
    
    try:
        data = await request.json()
        messages = data.get("messages", [])

        # ✅ Proper user-assistant pair extraction
        chat_memory.clear()
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                chat_memory.append((user_msg["content"], assistant_msg["content"]))

        if len(chat_memory) > MAX_HISTORY_LENGTH:
            summary = summarize_history(chat_memory)
            chat_log = f"Previous summary: {summary}"
            chat_memory.clear()
        else:
            chat_log = "\n".join(f"User: {q}\nDevBuddy: {a}" for q, a in chat_memory)

        latest_question = messages[-1]["content"]
        response = rag_chain.invoke({
            "input": f"{chat_log}\nUser: {latest_question}",
            "chat_history": chat_log
        })

        answer = response["answer"]
        chat_memory.append((latest_question, answer))

        answer = response["answer"]
        if not any(latest_question == q for q, _ in chat_memory):
            chat_memory.append((latest_question, answer))

        return {"role": "assistant", "content": answer}


    except Exception as e:
        traceback.print_exc()  # ✅ Print full error stack in console
        return JSONResponse(content={"error": str(e)}, status_code=500)


