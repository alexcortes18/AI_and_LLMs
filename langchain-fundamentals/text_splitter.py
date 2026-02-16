from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

textloader = TextLoader("./langchain-fundamentals/data/dream.txt")
documents = textloader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len, 
)

splits = text_splitter.split_documents(documents=documents)
for i, split in enumerate(splits):
    print(f"Split {i+1}:\n{split}\n")
print(type(splits[0]))