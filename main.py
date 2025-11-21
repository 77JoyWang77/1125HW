import aiofiles
from typing import Annotated
from typing_extensions import TypedDict

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from rag_pdf_groq import PDFRAGSystem
from react_weather_groq import WeatherAgent

from fastapi.middleware.cors import CORSMiddleware


class UserRequest(TypedDict):
    query: str
    files: list[UploadFile] | None


class ModelResponse(TypedDict):
    answer: str


app = FastAPI()
rag = PDFRAGSystem()
agent = WeatherAgent()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許的跨站來源
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法
    allow_headers=["*"],  # 允許所有 headers
)


async def save_file(path: str, file: UploadFile):
    async with aiofiles.open(path, "wb") as out:
        content = await file.read()
        await out.write(content)


@app.post("/api/ai/query")
async def ai_query(
    query: Annotated[str, Form()],
    files: Annotated[list[UploadFile] | None, File()] = None,
):
    print(query)
    print(files)
    if query.startswith("mode=rag"):  # rag
        query = query.lstrip("mode=rag")
        if files:  # rag 初始化
            for file in files:
                await save_file(f"./pdfs/{file.filename}", file)
            documents = rag.load_multiple_pdfs("./pdfs")
            if documents:
                rag.create_vector_store(documents)
            return {"answer": "好的"}
        else:  # rag 問答
            return {"answer": rag.query(query)}
    elif query.startswith("mode=weather"):
        query = query.lstrip("mode=weather")
        return {"answer": "好的"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
