from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import SeleniumURLLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# READ ABOUT CHAINS AT THE END!

from dotenv import load_dotenv

load_dotenv()
model_name = "gpt-4o-mini"

# List of documents to process
documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux/",
    "https://beebom.com/how-open-port-linux/",
    "https://beebom.com/linux-vs-windows/",
]

# SeleniumURLLoader uses a real browser via Selenium to fully render JavaScript-heavy web pages and returns the rendered page content
# as LangChain Document objects (text plus metadata like source URL) for downstream processing.
def scrape_docs(urls: List[str]) -> List[Dict]:
    """Scrape content from URLs using SeleniumURLLoader"""
    try:
        loader = SeleniumURLLoader(urls=urls)
        raw_docs = loader.load()
        print(f"\nSuccessfully loaded {len(raw_docs)} documents")

        # Print some information about the loaded documents
        for doc in raw_docs:
            print(f"\nSource: {doc.metadata.get('source', 'No source')}")
            print(f"Content length: {len(doc.page_content)} characters")

        return raw_docs

    except Exception as e:
        print(f"Error during document loading: {str(e)}")
        return []
    
def split_documents(pages_content: List[Dict]) -> tuple:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for document in pages_content:
        # Extract text from Document object
        text = document.page_content  # Changed from document to document.page_content
        source = document.metadata.get("source", "")  # Get source from metadata

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    print(f"Created {len(all_texts)} chunks of text")
    return all_texts, all_metadatas

def create_vector_store(texts: List[str], metadatas: List[Dict]):
    """ Create vector store (database) using ChromaDB"""
    embeddings = OpenAIEmbeddings(
         model="text-embedding-3-small"
    )
    db = Chroma.from_texts(texts = texts, metadatas=metadatas, embedding=embeddings)
    return db

def setup_qa_chain(db):
    """ Set up QA chain with polite response template"""
    llm = ChatOpenAI(model= model_name, temperature=0)
    retriever = db.as_retriever()
    
    # Create a custom prompt template
    prompt = ChatPromptTemplate.from_template(
        """
    Please provide a polite and helpful response to the following question, utilizing the provided context. 
    Ensure that the tone remains professional, courteous, and empathetic, and tailor your response to directly address the inquiry. 

    ### Context:
    {context}

    ### Question: 
    {question}

    ### Polite Response:
    In your response, consider including:
    - Acknowledge the user’s query and express gratitude for the opportunity to assist.
    - Provide a clear and concise answer that directly addresses the question.
    - Use positive language and maintain a supportive tone throughout.
    - If applicable, include relevant information or resources that could help further.
    - Conclude by inviting any follow-up questions or providing encouragement for the user’s pursuit of information.
        """
    )
    
    # Create the chain. The chain is a RunnableSequence and it can be ".invoke()"
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} # these both parts of the dict receive the input of the chain when
        # the chain is invoked: chain.invoke(query). Query is passed through {"context": retriever.invoke(query), "question": query}
        # and that new dict is returned.
        | prompt # the new prompt is a type 'PromptValue' with this new variables inside its text = {"context": retriever.invoke(query), "question": query}
        | llm # the prompt goes into the llm (ChatOpenAI runnable) and output is an AIMessage.
        | StrOutputParser() # the model's output (AIMessage) is extracted as a string.
    )

    return chain, retriever  # Return both chain and retriever

def process_query(chain_and_retriever, query: str):
    """ Process a query and return response"""
    try:
        chain, retriever = chain_and_retriever
        response = chain.invoke(query)
        
        # Get the sources from metadata using the retriever
        docs = retriever.invoke(query)
        sources_str = ", ".join([doc.metadata.get("source","") for doc in docs])
        
        return {"answer":response, "sources":sources_str}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "answer": "I apologize, but I encountered an error while processing your question.",
            "sources": "",
        }

def main():

    # 1. Scrape documents
    print("Scraping documents...")
    pages_content = scrape_docs(documents)

    # 2. Split documents
    print("Splitting documents...")
    all_texts, all_metadatas = split_documents(pages_content)

    # 3. Create vector store
    print("Creating vector store...")
    db = create_vector_store(all_texts, all_metadatas)

    # 4. Set up QA chain
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(db)

    # 5. Interactive query loop
    print("\nReady for questions! (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your question: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            break

        result = process_query(qa_chain, query)

        print("\nResponse:")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"].split(","):
                print("- " + source.strip())


if __name__ == "__main__":
    main()
    
    
# RUNNABLE / CHAIN NOTES:
"""
Any object that implements LangChain's Runnable interface supports .invoke().

Common runnable objects:
- RunnableSequence: a | b | c
- RunnableParallel (dict runnables): {"x": a, "y": b}
- RunnablePassthrough / RunnableLambda
- ChatPromptTemplate
- Chat/LLM models (e.g., ChatOpenAI)
- Retrievers (e.g., VectorStoreRetriever)
- Output parsers (e.g., StrOutputParser)
- Many LCEL-based chains and tool wrappers

LCEL (LangChain Expression Language) is the pipe syntax used to compose steps:
chain = prompt | llm | StrOutputParser()

Why this is a chain:
- Each step is runnable-compatible.
- The | operator composes them into a RunnableSequence.
- The output of one step becomes the input to the next step.

Note:
- A chain can be invoked with a normal string input.
- But chain *steps* cannot be raw strings; they must be runnables (or runnable-coercible objects).
In this code, the dict runnable has runnable values in its key-value pairs.
"""
