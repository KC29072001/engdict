# import numpy as np
# from .database import get_index, get_product_reviews
# from .embeddings import generate_embedding
# from .models import SearchResult

# def search_products(query: str, k: int = 5) -> list[SearchResult]:
#     query_embedding = generate_embedding(query)
#     index = get_index()
#     product_reviews = get_product_reviews()

#     distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)

#     results = []
#     for i, idx in enumerate(indices[0]):
#         review = product_reviews[idx]
#         result = SearchResult(
#             product_name=review['product_name'],
#             similarity_score=float(1 - distances[0][i]),  # Convert distance to similarity
#             review_snippet=review['text'][:100] + "..."  # Truncate long reviews
#         )
#         results.append(result)

#     return results


# working
# import numpy as np
# from .database import get_index, get_product_reviews
# from .embeddings import generate_embedding
# from .models import SearchResult

# def search_products(query: str, k: int = 5) -> list[SearchResult]:
#     query_embedding = generate_embedding(query)
#     index = get_index()
#     product_reviews = get_product_reviews()

#     distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)

#     results = []
#     for i, idx in enumerate(indices[0]):
#         review = product_reviews[idx]
#         result = SearchResult(
#             product_name=review['product_name'],
#             similarity_score=float(1 - distances[0][i] / np.max(distances)),  # Normalize similarity score
#             review_snippet=review['text'][:100] + "..."  # Truncate long reviews
#         )
#         results.append(result)

#     return results



import numpy as np
from .database import get_index, get_product_reviews
from .embeddings import generate_embedding
from .models import SearchResult

def search_products(query: str, k: int = 5) -> list[SearchResult]:
    query_embedding = generate_embedding(query)
    index = get_index()
    product_reviews = get_product_reviews()

    # Increase k to get more initial results
    initial_k = k * 3
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), initial_k)

    results = []
    seen_products = set()
    for i, idx in enumerate(indices[0]):
        review = product_reviews[idx]
        product_name = review['product_name'].lower()
        
        # Check if the query is a substring of the product name
        if query.lower() in product_name and product_name not in seen_products:
            similarity_score = 1 - (distances[0][i] / np.max(distances))
            
            # Only include results with a similarity score above a threshold
            if similarity_score > 0.5:
                result = SearchResult(
                    product_name=review['product_name'],
                    similarity_score=float(similarity_score),
                    review_snippet=review['text'][:100] + "..."  # Truncate long reviews
                )
                results.append(result)
                seen_products.add(product_name)

        if len(results) == k:
            break

    # Sort results by similarity score
    results.sort(key=lambda x: x.similarity_score, reverse=True)

    return results