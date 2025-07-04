from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


documents = [
    "Kerala is known for its scenic backwaters, coconut palms, and high literacy rate.",
    "Rajasthan captivates with its desert landscapes, majestic forts, and vibrant folk traditions.",
    "Tamil Nadu boasts ancient temples, classical music, and a strong technology sector in Chennai.",
    "Punjab is famous for its fertile farmlands, bhangra dance, and rich culinary heritage.",
    "West Bengal celebrates literary excellence, artistic cinema, and grand Durga Puja festivities."
]


query = "deity  intruments tech"

doc_embeddings = model.encode(documents)
query_embedding = model.encode(query)


score = cosine_similarity([query_embedding], doc_embeddings)
max_ind = np.argmax(score)

print(max_ind)
print(documents[max_ind])