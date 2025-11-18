from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from PyPDF2 import PdfReader
import json
import io
import os
import uvicorn

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
load_dotenv()

app = FastAPI(title="PDF Q&A API")
app.mount("/demo", StaticFiles(directory="/static", html=True), name="demo")
app.mount("/auto", StaticFiles(directory="/static", html=True), name="auto")

WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

if os.getenv("MODEL_ID") is not None:
    MODEL_ID = os.getenv("MODEL_ID")

def init_model():
    credentials = Credentials(api_key=WATSONX_APIKEY, url=WATSONX_URL)
    params = {"temperature": 0.0}
    return ModelInference(
        model_id=MODEL_ID,
        params=params,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID
    )


class Question(BaseModel):
    question: str


class QuestionInput(BaseModel):
    damage_type: Optional[str] = None
    questions: List[str]


class Answer(BaseModel):
    question: str
    answer: str


def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file bytes."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")


def query_llm(pdf_text: str, questions: List[str], damage_type: Optional[str] = None) -> List[Answer]:
    """Query Watson LLM with PDF context and questions."""
    
    # Construct the prompt
    questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    damage_context = f"\nDAMAGE TYPE: {damage_type}\n" if damage_type else ""
    
    prompt = f"""You are analyzing a PDF document related to insurance policy coverage. Based on the content provided, answer the following questions accurately and concisely.
{damage_context}
PDF CONTENT:
{pdf_text}

QUESTIONS:
{questions_formatted}

Provide your answers in valid JSON format as an array of objects, where each object has "question" and "answer" fields. 
Return ONLY the JSON array, no other text, explanation, or markdown formatting.
DO NOT number the questions.

Example format:
[
{{"question": "Question 1 text", "answer": "Answer to question 1"}},
{{"question": "Question 2 text", "answer": "Answer to question 2"}}
]"""

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Query the model
        model = init_model()
        response_text = model.chat(messages=messages)
        
        if not response_text or "choices" not in response_text or not response_text["choices"]:
            raise HTTPException(status_code=500, detail="Empty model response.")

        # Extract content
        content = response_text["choices"][0]["message"]["content"].strip()
        
        # Clean up potential markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON response
        answers_json = json.loads(content)
        
        # Validate it's a list
        if not isinstance(answers_json, list):
            raise ValueError("Response is not a JSON array")
        
        # Convert to Answer objects
        answers = [Answer(**item) for item in answers_json]
        return answers
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error parsing LLM response as JSON: {str(e)}. Response was: {content[:200]}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying LLM: {str(e)}")

@app.post("/api/upload")
async def upload_file(pdf_file: UploadFile = File(...)):
    """
    Upload PDF file to act as source for demo
    
    Parameters:
    - pdf_file: File to be used as source for all RAG
    
    Returns:
    - JSON {msg: 'Success'} 
    """

    try:
        # Check if a file is uploaded
        if not pdf_file:
            raise HTTPException(status_code=400, detail="No file part in the request")
        
        print("Received Filename:", pdf_file.filename)
        
        # Validate PDF file
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read PDF file
        pdf_content = await pdf_file.read()
        
        # Store the raw file and the contents in case needed later
        app.state.pdf_file = pdf_file
        app.state.pdf_content = pdf_content

        return JSONResponse(content={"msg": f"File '{pdf_file.filename}' uploaded and processed successfully", "success": True})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed. Error: {str(e)}")        

@app.post("/api/pdf-qa", response_model=List[Answer])
async def pdf_question_answer(questions: str = File(...)):
    """
    Upload questions to get answers.
    
    Parameters:
    - questions: JSON string containing array of questions or object with 'questions' array
    
    Returns:
    - JSON array of question-answer pairs
    """
    pdf_content = app.state.pdf_content
    
    # Validate PDF file
    if not pdf_content:
        raise HTTPException(status_code=400, detail="You did not select a policy yet")
    
    # Parse questions JSON
    try:
        questions_data = json.loads(questions)
        
        # Handle both formats
        if isinstance(questions_data, dict):
            if 'questions' in questions_data:
                questions_list = questions_data['questions']
                damage_type = questions_data.get('damage_type')
            else:
                raise ValueError("JSON object must contain 'questions' field")
        elif isinstance(questions_data, list):
            questions_list = questions_data
            damage_type = None
        else:
            raise ValueError("Invalid questions format - must be array or object with 'questions' field")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for questions")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not questions_list:
        raise HTTPException(status_code=400, detail="Questions array cannot be empty")
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_content)
    
    if not pdf_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Query LLM
    answers = query_llm(pdf_text, questions_list, damage_type)
    
    return answers

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
