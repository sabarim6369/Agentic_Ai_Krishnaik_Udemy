from langchain_community.document_loaders import TextLoader

loader = TextLoader("file.txt")  # Make sure file.txt exists too!
docs = loader.load()

print(docs[0].page_content)
