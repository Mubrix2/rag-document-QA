from loaders import load_documents
from chunking import chunk_documents
from embeddings import get_embeddings
from vectorstore import build_vectorstore
from retriever import get_retriever
from chains import build_qa_chain

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def answer_question(question):
    # 1. Load documents
    docs = load_documents("data/documents/RAG_Hands_On_Practical_Document.pdf")

    if not docs:
        return "I don't know. The information is not available in the documents."

    # 2. Chunk documents
    chunks = chunk_documents(docs)

    # 3. Embeddings + Vector store
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    # 4. Load open-source LLM (Hugging Face)
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 5. Build RAG chain
    qa_chain = build_qa_chain(llm, retriever)

    # 6. Ask question
    result = qa_chain(question)
    return result["result"]


# Usage
if __name__ == "__main__":
    answer = answer_question("What is the refund policy?")
    print(answer)