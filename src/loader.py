from langchain_community.document_loaders import TextLoader

path = "data\documents\RAG_Hands_On_Practical_Document.pdf"

def load_documents(path):
    loader = TextLoader(path)
    return loader.load()
