import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt

# create a chroma object to persist data
chroma_client = chromadb.PersistentClient(path="./Alex/Multimodal RAG/data/chroma.db")

#instantiate Image Loader
image_loader = ImageLoader()

#instantiate the multimodel embedding function
embedding_function = OpenCLIPEmbeddingFunction()

#create the collection - vector database
collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function= embedding_function,
    data_loader= image_loader
)

# add images to the collection add() or update() method
collection.add(
    ids=["0", "1"],
    uris=["./Alex/Multimodal RAG/images/lion.jpg", "./Alex/Multimodal RAG/images/tiger.jpg"],
    metadatas=[{"category": "animal"}, {"category": "animal"}],  # metadata - optional
)

print(collection.count())