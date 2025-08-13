from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__=="__main__":

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_KEY"]  # Optional if in env variable
    )

    llm = ChatGroq(model="llama3-8b-8192")

    vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)

    # with help of langchain-ai/retrieval-qa-chat

    # retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # combine_docs_chain = create_stuff_documents_chain(llm,retrival_qa_chat_prompt)

    # retrival_chain = create_retrieval_chain(retriever=vectorStore.as_retriever(), combine_docs_chain=combine_docs_chain)

    # query = "name any one cool project in langchain?"

    # result = retrival_chain.invoke(input={"input":query}) 

    # parser = StrOutputParser()

    # print(parser.parse(result["answer"]))

    custom_rag_template = """
    Use the context below to answer the question in the end. Dont try to make up the answer if you 
    dont have the knowledge on it. Answer in min 2 to 3 lines and max 5 to 6 lines. Try to answer in first person.At the end say thanks for asking!.

    context: {context}

    question: {question}

    helpfull answer: ''''''
    """

    rag_chain = (
        {"context":vectorStore.as_retriever() | format_docs, "question":RunnablePassthrough()} | PromptTemplate.from_template(custom_rag_template) | llm | StrOutputParser()
    )

    st.title("SRKGPT-2.0 - Langchain powered RAG Tool")
    user_question = st.text_area("Ask away any question about me")

    if st.button("Get Answer"):
        if user_question.strip():
            try:
                answer = rag_chain.invoke(user_question)
                st.info(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question to get an answer.")

    


