import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_indx_name = "MyFaissIndex.faiss"
d = 384 # diametion of all-MiniLM-L6-v2 embeddings

def creating_and_insert_to_faiss_index():
   data_for_faiss = [
     "data1",
     "data2",
     "data3"
   ]

   if len(data_for_faiss)>0:
        # Generate embeddings
        embeddings = model.encode(data_for_faiss,show_progress_bar=True)

        # Adding to embedding array
        embeddings = np.array(embeddings).astype('float32')

        # Adding arrray of embeddings to Index
        index.add(embeddings) # type: ignore

        # Write Index to Dist as file
        faiss.write_index(index, faiss_indx_name)
   else:
        print("Data is empty")

if os.path.exists(faiss_indx_name):
  print("Index Alreaddy Exists")
  exit()
else:
  print("Index Not Exists. Creating new index")
  index = faiss.IndexFlatL2(d)

creating_and_insert_to_faiss_index()
