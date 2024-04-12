import uvicorn
from fastapi import FastAPI
import joblib
from typing import List  # type: ignore
from model import ChatModel
import re  # type: ignore
import rag_util

app = FastAPI()


@app.get('/')
def index():
    return {'message': 'NER API'}


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
    return model


model = load_model()


@app.post("/paraphrase")
async def paraphrase(question: List[str]):
    print("QUESTION:", question)
    para_q_o = model.generate(question=question)
    para_q = preprocess_paraphrased_question(para_q_o, question)

    return {"paraphrased_question": para_q,
            "question": question,
            "non_regex_para": para_q_o,
            }

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')
