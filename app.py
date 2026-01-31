from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import re
import os
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Environment variable OPENAI_API_KEY is missing.")

# Initialize tiktoken encoding
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))



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
    model: str
    user_id: str = "default_user"
    session_id: str = "default_session"


class SentimentRequest(BaseModel):
    responses: list[str]
    model: str
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

    def call_openai(self, prompt: str, model_name: str):
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)



            response = client.chat.completions.create(
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



    def call_openai(self, prompt: str, model_name: str):
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)



            response = client.chat.completions.create(
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
def read_root():
    return {
        "message": "Welcome to the AI Survey Generator API!",
        "usage": "Send a POST request to /generate-question with a JSON body.",
        "example_body": {
            "query": "10 questions about python",
            "model": "gpt-4.1-mini"
        }

    }

@app.post("/generate-question")
def generate_question(request: RequestModel):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    generator = SurveyGenerator()

    num_questions = generator.extract_number(request.query)
    topic = generator.extract_topic(request.query)

    prompt = generator.build_prompt(topic, num_questions)

    response_json, full_prompt, raw_content = generator.call_openai(prompt, request.model)

    # cleanup
    for q in response_json.get("questions", []):
        q["text"] = generator.enforce_limit(generator.clean_text(q["text"]))

    # Count tokens
    input_tokens = count_tokens(full_prompt)
    output_tokens = count_tokens(raw_content)

    return {
        "status": "success",
        "user_id": request.user_id,
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


@app.post("/analyze-sentiment")
def analyze_sentiment(request: SentimentRequest):
    if not request.responses:
        raise HTTPException(status_code=400, detail="Responses list cannot be empty.")

    analyzer = SentimentAnalyzer()
    prompt = analyzer.build_prompt(request.responses, request.analysis_type, request.custom_prompt)
    result, full_prompt, raw_content = analyzer.call_openai(prompt, request.model)

    # Count tokens
    input_tokens = count_tokens(full_prompt)
    output_tokens = count_tokens(raw_content)

    response_data = {
        "status": "success",
        "user_id": request.user_id,
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

    return response_data




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
