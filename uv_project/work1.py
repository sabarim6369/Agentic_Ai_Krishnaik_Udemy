from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
import requests

# Load text and PDF files
loader = PyPDFLoader("pdf.pdf")
docs = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

# Embedding
access_token = "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"
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": access_token}
)
vectors = embedding.embed_documents([doc.page_content for doc in split_docs])

# Create FAISS index
vecs = np.array(vectors).astype('float32')
d = vecs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(vecs)
print(f"Number of vectors in index: {index.ntotal}")

# Query
query = "What is the main topic of the document?"
query_embedding = embedding.embed_query(query)
D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)

print("Distances:", D)
print("Indices:", I)

# Collect relevant chunks from retrieved indices
relevant_chunks = [split_docs[idx].page_content for idx in I[0]]
print("\nRelevant chunks retrieved:")
for chunk in relevant_chunks:
    print(chunk)
    print("-" * 40)

# Prepare prompt for LLM with retrieved context
context_text = "\n\n".join(relevant_chunks)
prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"

# Groq API call function
def call_groq_api(prompt):
    API_KEY = "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"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Call Groq API with context + question prompt
try:
    answer = call_groq_api(prompt)
    print("\nLLM Response:\n", answer)
except Exception as e:
    print("Error calling Groq API:", e)
