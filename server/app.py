from fastapi import FastAPI, Query

import main

app = FastAPI()

model = main.AIModel()

@app.get("/")
def root():
    return {"message": "connection online"}

# OpenAI API와 ChromaDB를 활용한 추가 엔드포인트 예시
@app.get("/recommend")
def recommend_bakery(prompt: str = Query()):
    return model.request(prompt)
