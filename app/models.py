from pydantic import BaseModel

class SearchResult(BaseModel):
    product_name: str
    similarity_score: float
    review_snippet: str


