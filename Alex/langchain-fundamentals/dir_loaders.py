# from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import(
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader
)
import os

print(os.getcwd())

dir_loader = DirectoryLoader("langchain-fundamentals/data/", glob="**/*.txt")
dir_documents = dir_loader.load()

# print("Directory Text Documents:", dir_documents)

for id, document in enumerate(dir_documents):
    print(f"Directory Text Documents {id}: {dir_documents} \n")