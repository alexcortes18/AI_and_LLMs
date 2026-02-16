from langchain_community.document_loaders import(
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader
)

pdf_loader = PyPDFLoader("langchain-fundamentals/doc/linux-manual.pdf")
doc = pdf_loader.load()
print("PDF Documents: ", doc)
