# llm work
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



# import numpy as np
# from .database import get_index, get_product_reviews
# from .embeddings import generate_embedding
# from .models import SearchResult

# def search_products(query: str, k: int = 5) -> list[SearchResult]:
#     query_embedding = generate_embedding(query)
#     index = get_index()
#     product_reviews = get_product_reviews()

#     # Increase k to get more initial results
#     initial_k = k * 3
#     distances, indices = index.search(np.array([query_embedding]).astype('float32'), initial_k)

#     results = []
#     seen_products = set()
#     for i, idx in enumerate(indices[0]):
#         review = product_reviews[idx]
#         product_name = review['product_name'].lower()
        
#         # Check if the query is a substring of the product name
#         if query.lower() in product_name and product_name not in seen_products:
#             similarity_score = 1 - (distances[0][i] / np.max(distances))
            
#             # Only include results with a similarity score above a threshold
#             if similarity_score > 0.5:
#                 result = SearchResult(
#                     product_name=review['product_name'],
#                     similarity_score=float(similarity_score),
#                     review_snippet=review['text'][:100] + "..."  # Truncate long reviews
#                 )
#                 results.append(result)
#                 seen_products.add(product_name)

#         if len(results) == k:
#             break

#     # Sort results by similarity score
#     results.sort(key=lambda x: x.similarity_score, reverse=True)

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
    initial_k = min(k * 10, len(product_reviews))
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), initial_k)

    results = []
    for i, idx in enumerate(indices[0]):
        review = product_reviews[idx]
        product_name = review['product_name'].lower()
        query_lower = query.lower()
        
        # Check if any word in the query is a substring of the product name
        if any(q_word in product_name for q_word in query_lower.split()):
            similarity_score = 1 - (distances[0][i] / np.max(distances))
            
            result = SearchResult(
                product_name=review['product_name'],
                similarity_score=float(similarity_score),
                review_snippet=review['text'][:100] + "..."  # Truncate long reviews
            )
            results.append(result)

        if len(results) == k:
            break

    # Sort results by similarity score
    results.sort(key=lambda x: x.similarity_score, reverse=True)

    # If no results, return the top k results regardless of name matching
    if not results:
        for i in range(min(k, len(indices[0]))):
            idx = indices[0][i]
            review = product_reviews[idx]
            similarity_score = 1 - (distances[0][i] / np.max(distances))
            result = SearchResult(
                product_name=review['product_name'],
                similarity_score=float(similarity_score),
                review_snippet=review['text'][:100] + "..."
            )
            results.append(result)

    return results

# # Add this function for debugging
# def debug_search(query: str):
#     query_embedding = generate_embedding(query)
#     index = get_index()
#     product_reviews = get_product_reviews()

#     k = 10  # Number of results to fetch for debugging
#     distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)

#     debug_info = []
#     for i, idx in enumerate(indices[0]):
#         review = product_reviews[idx]
#         debug_info.append({
#             'product_name': review['product_name'],
#             'distance': float(distances[0][i]),
#             'similarity_score': float(1 - (distances[0][i] / np.max(distances))),
#             'review_snippet': review['text'][:100] + "..."
#         })

#     return debug_info