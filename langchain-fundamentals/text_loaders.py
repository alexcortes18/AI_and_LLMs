import os
# Workaround for macOS OpenMP runtime conflict (duplicate libomp) triggered by native deps (e.g., FAISS).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint
import re
from dotenv import load_dotenv

load_dotenv()

# # We need to use "langchain-fundamentals" in the path since using "./" resolves to the working directory.
# # Relative paths are resolved from the process current working directory (cwd) find it as: print(os.getcwd()), 
# # not from the script file location.
# documents = TextLoader("./langchain-fundamentals/doc/dream.txt").load()
# # print(documents[:10])

# def clean_text(text):
#     # Remove unwanted characters (e.g., digits, special characters)
#     text = re.sub(r"[^a-zA-Z\s]", "", text)

#     # Normalize whitespace
#     text = re.sub(r"\s+", " ", text).strip()

#     # Convert to lowercase
#     text = text.lower()

#     return text

# # In here we clean a whole document (or documents if we had them) -> and receive list of clean strings of text
# clean_documents = [clean_text(doc.page_content) for doc in documents]
# # print(clean_documents)

# # In here we split the document into texts and apply the clean function to each text/chunk
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
# texts = text_splitter.split_documents(documents)
# texts = [clean_text(text.page_content) for text in texts]
# # print(texts)


# # Load OpenAI embeddings to vectorize the text
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # Create the retriever from the loaded embeddings and documents
# retriever = FAISS.from_texts(texts, embeddings).as_retriever(
#     search_kwargs = {"k":5}
#     ) # without .as_retriever() we get a vector store, but we want a retriever to call .invoke()
# # More explicit:
# # Without as_retriever(): you get a FAISS vector store object.
# # With as_retriever(): you get a VectorStoreRetriever wrapper (invoke, chain-friendly API).

# query = "What did Martin Luther King jr. dream about?"
# doc = retriever.invoke(query)
# pprint.pprint(f" => DOCS: {doc}:")


######################################################################################################
# Video tutorial is ok but can be better:
"""
Your current code works, but it drops Document objects too early and loses metadata.

What’s wrong in current flow

texts = [clean_text(text.page_content) for text in texts] converts chunks to plain strings.
Then FAISS.from_texts(...) indexes only strings, no metadata.
If you ever accidentally stringify full Document objects, you can embed noise (Document(...) repr) instead of clean content.
Better pattern

Load Documents.
Clean doc.page_content but keep Document shape.
Split into chunk Documents.
Index with FAISS.from_documents(...).
    """



import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

docs = TextLoader("./langchain-fundamentals/doc/dream.txt").load()

# Keep Documents, only clean page_content
clean_docs = [
    Document(page_content=clean_text(d.page_content), metadata=d.metadata)
    for d in docs
]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunk_docs = splitter.split_documents(clean_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunk_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# query = "What did Martin Luther King Jr. dream about?"
query = "Give me a summary of the speech in bullet points"
results = retriever.invoke(query)
pprint.pprint(results)
print("\n")

# Chat with model and our docs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Please use the following {docs} and answer the following question: {query}"
)
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

# Before sending the retrieve docs (results), we can format them to just send the page_content from them, and not the metadata
context = "\n\n".join(d.page_content for d in results)

print("Context sent to model:\n")
print(context)
print(f"\nChars:  {len(context)} \n")

response = chain.invoke({"docs": context, "query": query})
print(response)