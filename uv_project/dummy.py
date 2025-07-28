
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

# Load text and PDF files
loader = TextLoader("file.txt")
pdfloader = PyPDFLoader("pdf.pdf")

docs = loader.load()
docs_pdf = pdfloader.load()

# Combine both sources into one list
all_docs = docs + docs_pdf

# Use RecursiveCharacterTextSplitter to chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Adjust size as needed
    chunk_overlap=100     # Slight overlap for context continuity
)
split_docs = text_splitter.split_documents(all_docs)

for i,doc in enumerate(split_docs):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print("-" * 40)  # Separator for clarity