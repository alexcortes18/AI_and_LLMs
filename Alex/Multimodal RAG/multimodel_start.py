import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
import warnings

# create a chroma object to persist data
chroma_client = chromadb.PersistentClient(path="./Alex/Multimodal RAG/data/chroma.db")

#instantiate Image Loader
image_loader = ImageLoader()

#instantiate the multimodel embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# chroma_client.delete_collection("multimodal_collection")
#create the collection - vector database
collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function= embedding_function,
    data_loader= image_loader
)

# Use .add() to add a new record or .update() to update existing record
# on first run add() is used, on subsequent runs update() is used
collection.update(
    # ids=["0", "1"],
    # uris=["./Alex/Multimodal RAG/images/lion.jpg", "./Alex/Multimodal RAG/images/tiger.jpg"],
    # metadatas=[{"category": "animal"}, {"category": "animal"}],  # metadata - optional
    ids=["E23","E25","E33",],
    uris=[
        "./Alex/Multimodal RAG/images/E23-2.jpg",
        "./Alex/Multimodal RAG/images/E25-2.jpg",
        "./Alex/Multimodal RAG/images/E33-2.jpg",
    ],
    metadatas=[
        {
            "item_id": "E23",
            "category": "food",
            "item_name": "Braised Fried Tofu with Greens",
        },
        {
            "item_id": "E25",
            "category": "food",
            "item_name": "Sauteed Assorted Vegetables",
        },
        {"item_id": "E33", "category": "food", "item_name": "Kung Pao Tofu"},
    ],
)

# print(collection.count())

# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict):
    results_count = len(query_results["ids"][0])
    print(f"The results count is {results_count}")
    
    for i in range(len(query_list)):
        print(f"Printing the results of query {i} - {query_list[i]}:")
        
        for j in range(results_count):
            ids = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uris = query_results["uris"][i][j]
            
            print(f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}")
            
            print(f"data: {uris}")
            plt.imshow(data)
            plt.axis = "off"
            plt.show()
    
    
query_texts = ["food with carrots", "show me a picture of a lion"]

query_results = collection.query(
    query_texts= query_texts,
    n_results= 1,
    include= ["documents", "metadatas", "distances", "data", "uris"]
)

print_query_results(query_list = query_texts, query_results=query_results)