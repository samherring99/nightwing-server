from datamodel.util import ChatMessage, ChatRequest, ChatResponse, CompletionRequest, CompletionResponse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import asyncio
import torch

app = FastAPI()

model_path = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

model.cuda()
model.eval()

tokenizer.pad_token = tokenizer.eos_token

request_queue = asyncio.Queue()

def format_request(request) -> str:
    return request.messages[0].content

async def process_requests():
    while True:
        request_data = await request_queue.get()
        request, response_future = request_data["request"], request_data["response_future"]

        try:
            text = format_request(request)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=request.max_tokens,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            reply_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_future.set_result({"reply": reply_text})

        except Exception as e:
            print(f"Error in chat processing: {e}")
            response_future.set_result({"reply": f"Error: {str(e)}"})
        finally:
            request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_requests())

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response_future = asyncio.get_running_loop().create_future()
    await request_queue.put({"request": request, "response_future": response_future})
    response = await response_future
    return response

@app.post("/completion", response_model=CompletionResponse)
async def completion(request: CompletionRequest):
    response_future = asyncio.get_running_loop().create_future()
    await request_queue.put({"request": request, "response_future": response_future})
    response = await response_future
    return response