from loaders import load_documents
from chunking import chunk_documents
from embeddings import get_embeddings
from vectorstore import build_vectorstore
from retriever import get_retriever
from chains import build_qa_chain
from langchain.llms import OpenAI

docs = load_documents("data/documents/company_policy.txt")
chunks = chunk_documents(docs)
embeddings = get_embeddings()
vectorstore = build_vectorstore(chunks, embeddings)
retriever = get_retriever(vectorstore)

llm = OpenAI()
qa_chain = build_qa_chain(llm, retriever)

result = qa_chain("What is the refund policy?")
print(result["result"])
