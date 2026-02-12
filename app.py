from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import re
import os
import tiktoken
import requests
import logging
from typing import Optional
from fastapi import Header, Depends, Body
from dotenv import load_dotenv
import motor.motor_asyncio
from datetime import datetime

# Load environment variables
load_dotenv()

# Get API key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Environment variable OPENAI_API_KEY is missing.")

# Get default model from env
DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# Initialize tiktoken encoding
encoding = tiktoken.get_encoding("cl100k_base")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authentication API URL
AUTH_API_URL = "http://10.199.207.155:8083/jwt-0.0.1-SNAPSHOT/api/protected"

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "SurveyAI question")

# Initialize MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
mongo_db = client[MONGO_DB_NAME]

# Database for User Management and Result Storage
class Database:
    def __init__(self):
        self.users = mongo_db["users"]
        self.survey_results = mongo_db["survey_results"]
        self.sentiment_results = mongo_db["sentiment_results"]

    async def get_or_create_user(self, username: str):
        user = await self.users.find_one({"username": username})
        if not user:
            user = {
                "username": username,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
            await self.users.insert_one(user)
        else:
            await self.users.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        return user

    async def save_survey(self, data: dict):
        data["timestamp"] = datetime.utcnow()
        await self.survey_results.insert_one(data)

    async def save_sentiment(self, data: dict):
        data["timestamp"] = datetime.utcnow()
        await self.sentiment_results.insert_one(data)

db = Database()

# ================= JWT Authentication Dependency =================
async def token_required(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Access token is missing!")
    
    try:
        # Validate token with external API
        # Note: 'authorization' should be the full value including "Bearer " if required by the service
        response = requests.get(AUTH_API_URL, headers={"Authorization": authorization})
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        # Extract username from API response
        # Format: "Hello, jyoti.raut@softelnetworks.com! This is a protected API."
        match = re.search(r"Hello, (.+?)! This is a protected API", response.text)
        if not match:
            logger.error(f"Unexpected API response format: {response.text}")
            raise HTTPException(status_code=401, detail="Failed to parse identity from auth service")
        
        username = match.group(1)

        # Get or create local user record
        user = await db.get_or_create_user(username)
        return user
        
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise HTTPException(status_code=401, detail="Token validation failed")


app = FastAPI(title="AI Survey Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.199.207.207:8081", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Request Schema ----------------------
class RequestModel(BaseModel):
    query: str
    model: Optional[str] = DEFAULT_MODEL
    user_id: str = "default_user"
    session_id: str = "default_session"


class SentimentRequest(BaseModel):
    responses: list[str]
    model: Optional[str] = DEFAULT_MODEL
    analysis_type: str = "count"
    custom_prompt: str = None
    user_id: str = "default_user"
    session_id: str = "default_session"





# ---------------------- Survey Generator Logic ----------------------
class SurveyGenerator:

    def extract_number(self, text: str) -> int:
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 10

    def extract_topic(self, text: str) -> str:
        clean = re.sub(r'\d+', '', text)
        clean = clean.replace("questions", "").replace("question", "").replace("create", "")
        return clean.strip() or "General Survey"

    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def enforce_limit(self, text: str, limit: int = 20) -> str:
        words = text.split()
        return text if len(words) <= limit else " ".join(words[:limit]) + "..."

    def build_prompt(self, topic: str, count: int) -> str:
        return f"""
        You must return ONLY valid JSON. No commentary.

        Create exactly {count} multiple-choice survey questions about "{topic}".

        Requirements:
        - Every question MUST be Multiple Choice
        - Exactly 4 options: A), B), C), D)
        - Max 20 words per question
        - No yes/no, no rating scale, no open-ended

        JSON format:
        {{
            "topic": "{topic}",
            "total_questions": {count},
            "questions": [
                {{
                    "id": 1,
                    "text": "example question",
                    "options": ["A)...","B)...","C)...","D)..."]
                }}
            ]
        }}
        """

    async def call_openai(self, prompt: str, model_name: str):
        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.6
            )

            content = response.choices[0].message.content.strip()

            # Extract valid JSON if extra text appears
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)

            return json.loads(content), prompt, content

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")



class SentimentAnalyzer:
    def build_prompt(self, responses: list[str], analysis_type: str = "count", custom_prompt: str = None) -> str:
        if custom_prompt:
            return f"""
            {custom_prompt}

            Responses:
            {json.dumps(responses)}
            """
        elif analysis_type == "summary":
            return f"""
            You must return ONLY valid JSON. No commentary.

            Analyze the sentiment of the following survey responses.
            Provide a natural language summary of the overall sentiment and identify any key trends or flows in the feedback.

            Responses:
            {json.dumps(responses)}

            JSON format:
            {{
                "summary": "A detailed natural language summary of the sentiment...",
                "key_trends": ["Trend 1", "Trend 2"]
            }}
            """
        else:
            return f"""
            You must return ONLY valid JSON. No commentary.

            Analyze the sentiment of the following survey responses.
            Classify them into: Positive, Negative, Neutral.
            Return the count of each category.

            Responses:
            {json.dumps(responses)}

            JSON format:
            {{
                "positive": 0,
                "negative": 0,
                "neutral": 0
            }}
            """



    async def call_openai(self, prompt: str, model_name: str):
        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.0
            )

            content = response.choices[0].message.content.strip()

            # Try to extract and parse JSON
            try:
                # Extract valid JSON if extra text appears
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    json_content = match.group(0)
                    return json.loads(json_content), prompt, content
                else:
                    # If no JSON structure found, try loading the whole content
                    return json.loads(content), prompt, content
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw content
                return content, prompt, content


        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")


# ---------------------- ENDPOINTS ----------------------

@app.get("/")
async def read_root(user: dict = Depends(token_required)):
    return {
        "message": "Welcome to the AI Survey Generator API!",
        "usage": "Send a POST request to /generate-question with a JSON body.",
        "example_body": {
            "query": "10 questions about python",
            "model": "gpt-4.1-mini"
        }

    }

@app.post("/generate-question")
async def generate_question(request: RequestModel, user: dict = Depends(token_required)):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    generator = SurveyGenerator()

    num_questions = generator.extract_number(request.query)
    topic = generator.extract_topic(request.query)

    prompt = generator.build_prompt(topic, num_questions)

    response_json, full_prompt, raw_content = await generator.call_openai(prompt, request.model)

    # cleanup
    for q in response_json.get("questions", []):
        q["text"] = generator.enforce_limit(generator.clean_text(q["text"]))

    # Count tokens
    input_tokens = count_tokens(full_prompt)
    output_tokens = count_tokens(raw_content)

    result_data = {
        "status": "success",
        "user_id": user.get("username", request.user_id),
        "session_id": request.session_id,
        "model_used": request.model,
        "topic_detected": topic,
        "question_count": num_questions,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        },
        "survey": response_json
    }

    # Save to MongoDB
    await db.save_survey(result_data.copy())

    return result_data


@app.post("/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest, user: dict = Depends(token_required)):
    if not request.responses:
        raise HTTPException(status_code=400, detail="Responses list cannot be empty.")

    analyzer = SentimentAnalyzer()
    prompt = analyzer.build_prompt(request.responses, request.analysis_type, request.custom_prompt)
    result, full_prompt, raw_content = await analyzer.call_openai(prompt, request.model)

    # Count tokens
    input_tokens = count_tokens(full_prompt)
    output_tokens = count_tokens(raw_content)

    response_data = {
        "status": "success",
        "user_id": user.get("username", request.user_id),
        "session_id": request.session_id,
        "model_used": request.model,
        "analysis_type": request.analysis_type,
        "total_responses": len(request.responses),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    }

    if request.custom_prompt:
        response_data["custom_analysis"] = result
    elif request.analysis_type == "summary":
        response_data["sentiment_summary"] = result
    else:
        response_data["sentiment_counts"] = result

    # Save to MongoDB
    await db.save_sentiment(response_data.copy())

    return response_data




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
