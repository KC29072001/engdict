from fastapi import FastAPI, Query
from .models import SearchResult
from .search import search_products

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Engineering Dictionary AI Agent"}

@app.get("/search", response_model=list[SearchResult])
async def search(query: str = Query(..., min_length=2)):
    results = search_products(query)
    return results