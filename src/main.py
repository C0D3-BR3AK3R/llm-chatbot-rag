import uvicorn
import re  # type: ignore
from typing import List  # type: ignore

from fastapi import FastAPI, Depends
from fastapi_health import health
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import ChatModel


class Question(BaseModel):
    question: str


app = FastAPI()

# Configure CORS

app.add_middleware(
    CORSMiddleware,
    # Change "*" to the appropriate origin URL of your ReactJS frontend
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

has_model_loaded = bool()


@app.get('/')
def index():
    return {'message': 'Paraphraser API'}


def preprocess_paraphrased_question(input_string, original_question):
    # Define the regex pattern to capture the introductory line and the question
    pattern = r"(?:.*?\b(?:paraphrased|rephrased|restated)\b.*?:\s*\"?)(.*)"

    # Search for the question using the regex pattern
    match = re.search(pattern, input_string, re.IGNORECASE)

    # Extract the question from the match
    if match:
        question = match.group(1)
        return question
    else:
        return original_question


def load_model():
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    global has_model_loaded
    has_model_loaded = True
    return model


def is_model_online(session: bool = has_model_loaded):
    # Ye script chalegi matlab model is online.
    return session


model = load_model()


@app.post("/paraphrase")
async def paraphrase(question: Question):
    print("QUESTION:", question)
    para_q_o = model.generate(question=question)
    para_q = preprocess_paraphrased_question(para_q_o, question)

    return {"paraphrased_question": para_q,
            "question": question,
            "non_regex_para": para_q_o,
            }


@app.post("/gemma-chat")
async def gemma_chat(question: Question):
    print("QUESTION:", question)
    ans = model.generate(question=question, mode='qna')
    print("ANSWER:", ans)
    return {
        "question": question,
        "answer": ans
    }


@app.get("/health")
def check_health(online_status: bool = has_model_loaded):
    """
    Endpoint to check the health status of the model.
    """
    if online_status:
        return {"status": "Model is online"}
    else:
        return {"status": "Model is offline"}


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')
