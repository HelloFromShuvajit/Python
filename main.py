import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/api")
def api_status():
    return {"message": "Grok FastAPI is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        print(f"Received message: {request.message}")
        print(f"API Key: {os.getenv('GROQ_API_KEY')[:10]}...")
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": request.message
                }
            ]
        )

        return {
            "response": response.choices[0].message.content
        }
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return {
            "response": f"Error: {str(e)}"
        }