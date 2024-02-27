import traceback
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_indx = "MyFaissIndex.faiss"
d = 384 # diametion of all-MiniLM-L6-v2 embeddings


if os.path.exists(faiss_indx):
   print("Loading Existing Index")
   index = faiss.read_index(faiss_indx)
else:
   print("Index Not Exists")
   exit()

def search_in_index(query:str):
  query_embedding = model.encode(query)
  distances, results = faiss_index.search(np.array([query_embedding]), k=5)  # type: ignore
  print(distances)
  print(results)
