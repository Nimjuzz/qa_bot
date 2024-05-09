import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate

def load_environment_variables():
    load_dotenv()

def load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embeddings

def add_to_chroma(new_chunks, new_chunks_ids, chroma_path):
    db = Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function())
    db.add_documents(new_chunks, ids=new_chunks_ids)
    db.persist()

def get_context(question, chunks, chroma_path):
    embedding = get_embedding_function()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=chroma_path)

    # Automatically persisted, no need to call persist again
    vectordb = Chroma(persist_directory=chroma_path, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    docs = retriever.invoke(question)  # Directly get the documents without unpacking
    context_text = "\n".join(doc.page_content for doc in docs)
    return context_text

def main(question):
    load_environment_variables()
    data_path = os.getenv("DATA_PATH")
    chroma_path = os.getenv("CHROMA_PATH")

    documents = load_documents(data_path)
    chunks = split_documents(documents)
    context = get_context(question, chunks, chroma_path)

    openai_api_key = os.getenv("API_KEY")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
    print(openai_api_key)

    template = """
    You are a Q&A assistant for university students.
    Your goal is to answer questions as accurately as possible based on context provided.
    Context: {context}
    ---
    You must answer in Russian.
    If you don't have information to answer the question, type "К сожалению, я не владею такой информацией. Попробуйте обратиться к эдвайзеру."
    Question: {question}
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    result = (prompt | llm).invoke({"context": context, "question": question})
    return result


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/process_query/")
async def process_query(query: Query):
    response = main(query.question)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)