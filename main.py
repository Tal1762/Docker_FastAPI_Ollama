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

def image_component(pil_image):
    return {'type': 'image', 'image': pil_image}

def ask_ai(file: UploadFile=File(...)):
    model_id = 'HuggingFaceTB/SmolVLM-Instruct'
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name
        pil_image_lst = convert_from_path(temp_file_path)
    conversation = [{'role': 'user', 'content': [{'type': 'text', 'text':  'State the following metrics from the financial statement for the most recent year available in the financial statement. DO NOT use metrics from any year earlier than the most recent one. Format your response by placing the numbers after their respective labels given here: {total net sales: , net income: , total assets: }. Convert the metrics you find in the financial statement to millions of dollars if they are not already in millions of dollars. Do not put dollar symbols before the metrics and do not use any commas in the metrics. Put the words NOT FOUND if a metric is not in the financial statement.'}]}]
    for i in range(0, len(pil_image_lst)):
        conversation[0]['content'].insert(0, image_component(pil_image_lst[i]))
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print('GPU is available. The model will use the GPU.')
    else:
        print('GPU is not available. The model will use the CPU.')
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print('prompt loaded yessir')
    inputs = processor(text=prompt, images=pil_image_lst, return_tensors='pt').to(model.device)
    print('we getting there')
    output = model.generate(**inputs, temperature=0.7, top_p=0.9, max_new_tokens=25, do_sample=True)
    print('we got output boys')
    processor_output = processor.decode(output[0])[len(prompt):]
    output = processor_output[processor_output.rfind("Assistant: ")+11:processor_output.rfind("<end_of_utterance>")]
    return output
  
app = FastAPI()

async def do_rag(file: UploadFile=File(...)):
    os.environ["HF_TOKEN"] = "userdata.get('HF_TOKEN')"
    os.environ["ANTHROPIC_API_KEY"] = "userdata.get('ANTHROPIC_API_KEY')"

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

@app.post('/rag/')
async def rag(file: UploadFile=File(...)):
  return {'response': do_rag(file)}

@app.post('/ai/')
async def ask_question(file: UploadFile=File(...)):
  return {'response': ask_ai(file)}

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