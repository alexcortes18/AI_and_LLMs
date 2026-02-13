from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint
import re
from dotenv import load_dotenv

load_dotenv()

documents = TextLoader("./doc/dream.txt").load()

print(documents[:10])
