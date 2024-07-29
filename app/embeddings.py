# from transformers import AutoTokenizer, AutoModel
# import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("distilbert-base-uncased")

# # def generate_embedding(text: str):
# #     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
# def generate_embedding(text: str):
#     # Ensure the text is properly encoded
#     text = text.encode('utf-8', errors='ignore').decode('utf-8')
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# multi lingual model with updated def to handle utf-8
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
def generate_embedding(text: str):
    # Ensure the text is properly encoded
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()