import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ =="__main__":
    print("start ingesting......")
    loader = TextLoader("./srkinfo.txt")
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=10, separator="\n")
    texts = text_splitter.split_documents(document)
    #print(f"created {len(texts)}")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_KEY"]  # Optional if in env variable
    )
    PineconeVectorStore.from_documents(texts,embeddings,index_name = os.environ["INDEX_NAME"])
    print("finish")