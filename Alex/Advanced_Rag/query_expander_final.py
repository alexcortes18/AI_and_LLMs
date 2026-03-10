import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma  # Updated import
from chromadb.config import Settings
import shutil
import streamlit as st  # Optional for visualization

load_dotenv()