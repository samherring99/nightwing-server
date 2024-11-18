from datamodel.util import ChatMessage, ChatRequest, ChatResponse, EmbedRequest, EmbedResponse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import asyncio
import torch

app = FastAPI()

model_path = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
embed_model = AutoModel.from_pretrained("BAAI/bge-small-en")
embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")

model.cuda()
model.eval()
embed_model.eval()

tokenizer.pad_token = tokenizer.eos_token

chat_request_queue = asyncio.Queue()
embed_request_queue = asyncio.Queue()


def format_request(request) -> str:
    if request.use_chat_template:
        text = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    else:
        formatted_text = ""
        for msg in request.messages:
            role = msg.role.capitalize()
            content = msg.content
            formatted_text += f"{role}: {content}\n"
        return formatted_text.strip()

async def process_chat_requests():
    while True:
        request_data = await chat_request_queue.get()
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