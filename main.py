from datamodel.util import ChatMessage, ChatRequest, ChatResponse, EmbedRequest, EmbedResponse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import asyncio
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("solidrust/Mistral-7B-Instruct-v0.3-AWQ")
model = AutoModelForCausalLM.from_pretrained("solidrust/Mistral-7B-Instruct-v0.3-AWQ")
embed_model = AutoModel.from_pretrained("BAAI/bge-small-en")
embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")

model.cuda()
model.eval()
embed_model.eval()

tokenizer.pad_token = tokenizer.eos_token

chat_request_queue = asyncio.Queue()
embed_request_queue = asyncio.Queue()

async def process_chat_requests():
    while True:
        request_data = await chat_request_queue.get()
        request, response_future = request_data["request"], request_data["response_future"]

        prompt = " ".join([msg.content + "\n" for msg in request.messages])
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(),
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
            )

        reply_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:])[0]
        response_future.set_result({"reply": reply_text})
        chat_request_queue.task_done()

async def process_embed_requests():
    while True:
        request_data = await embed_request_queue.get()
        request, response_future = request_data["request"], request_data["response_future"]

        if not request.text.strip():
            response_future.set_result({"embeddings": []})
            embed_request_queue.task_done()
            continue

        inputs = embed_tokenizer(request.text, return_tensors="pt")
        
        try:
            with torch.no_grad():
                embedding = embed_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
                
            response_future.set_result({"embeddings": embedding})
        except Exception as e:
            print(f"Error during embedding processing: {e}")
            response_future.set_result({"embeddings": []})
        finally:
            embed_request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_chat_requests())
    asyncio.create_task(process_embed_requests())

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response_future = asyncio.get_running_loop().create_future()
    await chat_request_queue.put({"request": request, "response_future": response_future})
    response = await response_future
    return response

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    response_future = asyncio.get_running_loop().create_future()
    await embed_request_queue.put({"request": request, "response_future": response_future})
    response = await response_future
    return response