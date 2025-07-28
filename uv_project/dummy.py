
# from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# loader=TextLoader("file.txt")
# pdfloader=PyPDFLoader("pdf.pdf")
# # webloader=WebBaseLoader("https://leetcode.com/u/subhashini01/")
# docs = loader.load()
# docs_pdf = pdfloader.load()
# # docs_web = webloader.load()
# # print(docs[0].page_content)
# # print(docs_pdf[0].page_content)
# print(docs_web[0].page_content) 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load text and PDF files


##########Data Ingestion
loader = TextLoader("file.txt")
pdfloader = PyPDFLoader("pdf.pdf")

docs = loader.load()
docs_pdf = pdfloader.load()

# Combine both sources into one list
all_docs = docs + docs_pdf


##########Data Transformation

# Use RecursiveCharacterTextSplitter to chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Adjust size as needed
    chunk_overlap=100     # Slight overlap for context continuity
)
textsplitterfromcharactertextsplitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100
)
# htmlheadersplitter=HTMLHeaderTextSplitter(
#     headers_to_split_on=[
#         ("h1", "Header 1"),
#         ("h2", "Header 2"),
#         ("h3", "Header 3"),
#     ]
# )

split_docs = text_splitter.split_documents(all_docs)
split=textsplitterfromcharactertextsplitter.split_documents(all_docs)
# split_html=htmlheadersplitter.split_documents(all_docs)
# for i,doc in enumerate(split_docs):
#     print(f"Document {i+1}:")
#     print(doc.page_content)
#     print("-" * 40)  # Separator for clarity

# print(split);
# for i,doc in enumerate(split_docs):
#     print(f"Document {i+1}:")
#     print(doc.page_content)
#     print("-" * 40)  # Separator for clarity



# RecursiveJsonSplitter   ->for JSON files
# RecursiveCharacterTextSplitter   ->for text files
# CharacterTextSplitter   ->for text files
# HTMLHeaderTextSplitter   ->for HTML files
# CSVHeaderTextSplitter   ->for CSV files
# RecursiveMarkdownTextSplitter   ->for Markdown files
# RecursiveCodeTextSplitter   ->for code files



####### Data embedding
#sentence transformer model architecture
from langchain_huggingface import HuggingFaceEmbeddings

access_token = "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"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": access_token}
)
vectors=embeddings.embed_documents([doc.page_content for doc in split_docs])
print(vectors)

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# embeddings=OllamaEmbeddings()
# db=FAISS.from_documents(split_docs, embeddings)
# print(db);




#####  FAISS store
import faiss
import numpy as np
vecs= np.array(vectors).astype('float32')
d = vecs.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(d)  # L2 distance index (simple exact search)
index.add(vecs)
print(f"Number of vectors in index: {index.ntotal}")



####### Querying the FAISS index
query= "Lead management project explained"
query_embedding=embeddings.embed_query(query)
query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)  # shape (1, dimension)
k = 2  # number of nearest neighbors to retrieve
distances, indices = index.search(query_vector, k)  
print("Distances:", distances)
print("Indices:", indices)
for idx in indices[0]:
    print(f"Document chunk: {split_docs[idx].page_content}")
    print("-" * 40)