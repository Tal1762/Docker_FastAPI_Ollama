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
import torch
import io
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaProcessor, file_utils, AutoModelForVision2Seq
from pdf2image import convert_from_path
import base64
import os

model_id = "HuggingFaceTB/SmolVLM-Instruct"

model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16, cache_dir=r'D:\Software\llama3cache')
processor = AutoProcessor.from_pretrained(model_id, cache_dir=r'D:\Software\llama3cache')

pdf_file = r"statement.pdf"
pil_image_lst = convert_from_path(pdf_file) # This returns a list even for a 1 page pdf
pil_image = pil_image_lst[0]

conversation = [
    {
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": "Describe this image in two sentences of 30 words or less"},
        ],
    },
]
print(torch.version.cuda)
display(pil_image)
if torch.cuda.is_available():
    print("GPU is available. The model will use the GPU.")
else:
    print("GPU is not available. The model will use the CPU.")

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,tokenize=False)
print("prompt loaded yessir")
inputs = processor(text = prompt, images = pil_image, return_tensors="pt").to(model.device)
print("we getting there")
output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=512, early_stopping=True)
print("we got output boys")
print("text&image_output: ",processor.decode(output[0])[len(prompt):])
"""