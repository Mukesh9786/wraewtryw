from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json

def load_json_as_docs(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    docs = []
    for entry in raw_data:
        content = "\n".join(f"{k}: {v}" for k, v in entry.items())
        docs.append(Document(page_content=content))
    return docs

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def build_custom_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    data_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are a helpful assistant for analyzing structured policy request data.

Use the following context to answer the user's question. Respond in friendly sentences.

Context:
{context}

Question: {question}

Answer:
"""
    )

    data_chain = LLMChain(llm=llm, prompt=data_prompt)

    def custom_chain(question: str, pending_intent=None):
        q = question.strip().lower()

        greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
        if any(greet in q for greet in greetings):
            return "😊 👋 Hello! How can I assist you today?"

        # Follow-up confirmation logic
        if q == "yes" and pending_intent == "WRStatus_NOT_COMPLETED":
            results = vectorstore.similarity_search("WRStatus not COMPLETED", k=15)
            filtered = [doc for doc in results if "WRStatus: COMPLETED" not in doc.page_content]
            top_docs = "\n".join([doc.page_content for doc in filtered[:10]])
            return "📄 (Based on policy data)\n\nHere are the WorkRequestIds with WRStatus not COMPLETED:\n\n" + top_docs

        # Detect user's intent
        if "wrstatus" in q and "not completed" in q:
            # Ask user for confirmation
            return "🧐 Are you looking for WorkRequestIds where WRStatus is not COMPLETED? Please reply 'Yes' to confirm."

        # Default behavior: retrieve and respond
        results = vectorstore.similarity_search_with_score(question, k=10)
        threshold = 2.0
        relevant_docs = [doc for doc, score in results if score < threshold]

        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs[:10]])
            response = data_chain.invoke({"question": question, "context": context})
            return "📄 (Based on policy data)\n\n" + response["text"]
        else:
            fallback_prompt = PromptTemplate(
                input_variables=["question"],
                template="""
You are a helpful assistant. Respond to the user's question naturally.

Question: {question}
"""
            )
            fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)
            response = fallback_chain.invoke({"question": question})
            return "🤖 " + response["text"]

    return custom_chain
