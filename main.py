import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

app = FastAPI()

# Model for the API input
class PromptRequest(BaseModel):
    model: str = "llama3.2:1b"
    prompt: str
    file: bytes = ""

# Helper function to interact with ollama
async def generate_response(model: str, prompt: str) -> str:
    try:
        # Call ollama's chat function and stream the response
        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )

        response_text = ""
        # Collect the streamed content
        for chunk in stream:
            response_text += chunk['message']['content']

        return response_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

def process_pdf(pdf_path):
    """Process the PDF, split it into chunks, and return the chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    document_text = "".join([page.page_content for page in pages])
    print([page.page_content for page in pages])
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Adjust as needed
        chunk_overlap=40  # Adjust as needed
    )
    chunks = text_splitter.create_documents([document_text])

    return chunks

@app.post("/file/")
async def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name
    
    document_chunks = process_pdf(temp_file_path)
    print(type(file.file.read()))
    response = await generate_text(PromptRequest(prompt="Find the total net sales in this financial statement, do not respond with anything other than the number:" + document_chunks[0].page_content))
    return {"filename": file.filename, "response": response["generated_text"]}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    model = request.model
    prompt = request.prompt

    # Generate the response using the helper function
    response = await generate_response(model, prompt)

    return {"generated_text": response}

"""
# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /app/main.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 2025-02-27 23:22:32 UTC (1740698552)

import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
#from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import torch
import io
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaProcessor, file_utils, AutoModelForVision2Seq
from pdf2image import convert_from_path
import base64
import os
import replicate

os.environ['REPLICATE_API_TOKEN'] = "r8_8Q74pvqI5saxf4Rmz1UoLo6OwxuPwDx3yqK7A"

def ask_ai(file: UploadFile=File(...)):
    model_id = 'HuggingFaceTB/SmolVLM-Instruct'
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name
        pil_image_lst = convert_from_path(temp_file_path)
    pil_image = pil_image_lst[0]
    conversation = [{'role': 'user', 'content': [{'type': 'image', 'image': pil_image}, {'type': 'text', 'text': 'Describe this image in two sentences of 30 words or less'}]}]
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print('GPU is available. The model will use the GPU.')
    else:
        print('GPU is not available. The model will use the CPU.')
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print('prompt loaded yessir')
    inputs = processor(text=prompt, images=pil_image, return_tensors='pt').to(model.device)
    print('we getting there')
    output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=25, do_sample=True)
    print('we got output boys')
    print('text&image_output: ', processor.decode(output[0])[len(prompt):])
    return processor.decode(output[0])[len(prompt):]
  
app = FastAPI()

class PromptRequest(BaseModel):
    model: str = 'llama3.2:1b'
    prompt: str
    file: bytes = ''

async def generate_response(model: str, prompt: str) -> str:
    try:
        stream = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], stream=True)
        response_text = ''
        for chunk in stream:
            response_text += chunk['message']['content']
        return response_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating response: {e}')

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    document_text = ''.join([page.page_content for page in pages])
    print([page.page_content for page in pages])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = text_splitter.create_documents([document_text])
    return chunks

@app.post('/ai/')
async def ask_question(file: UploadFile=File(...)):
  output = replicate.run(
    "lucataco/smolvlm-instruct:e79f1e0eb64fe9a145d0a0afd6127d43b37de66eaaa2e00ff3d165bc14097dfb",
    input={
        "image": "https://replicate.delivery/pbxt/M41uQ4M8J9FEqxRJ0tNnliJF2PNJIeGjdid66k2uHOLgv5OJ/weather.png",
        "prompt": "Where do the severe droughts happen according to this image?",
        "max_new_tokens": 500
    }
  )
  return {'response': output}

@app.post('/file/')
async def upload_file(file: UploadFile=File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name
    document_chunks = process_pdf(temp_file_path)
    print(type(file.file.read()))
    response = await generate_text(PromptRequest(prompt='Find the total net sales in this financial statement, do not respond with anything other than the number:' + document_chunks[0].page_content))
    return {'filename': file.filename, 'response': response['generated_text']}

@app.post('/generate')
async def generate_text(request: PromptRequest):
    model = request.model
    prompt = request.prompt
    response = await generate_response(model, prompt)
    return {'generated_text': response}
"""