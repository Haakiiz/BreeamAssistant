import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import pinecone
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import unidecode
from langchain_pinecone import PineconeVectorStore
import langchain_community


load_dotenv()




# Initialize Pinecone
pc = Pinecone(api_key="79928a79-af77-4ea0-9c1a-0c44153881e1")
# Check if the index already exists
existing_indexes = pc.list_indexes()
if "breeam-documents" not in [idx["name"] for idx in existing_indexes]:
    pc.create_index(
        name="breeam-documents",
        dimension=1536,  # Replace with your model dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )
else:
    print("Index already exists. Proceeding to use it...")



current_dir = os.path.dirname(__file__)
docs_dir = os.path.join(current_dir, "data")
index = pc.Index(os.getenv("PINECONE_INDEX"))
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")


def embedding_cost_calculator(chunks):
    model_cost = 0.0004 / 1000
    total_tokens = 0
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    for chunk in chunks:
        total_tokens += len(encoding.encode(chunk.page_content))
    cost = total_tokens * model_cost
    return f"{cost:.7f}"


def estimate_total_cost():
    total_estimated_cost = 0

    for root, dirs, files in os.walk(docs_dir):
        for filename in files:
            print(root, filename)
            if filename.endswith(".pdf"):  # Check if the file is a PDF
                file_path = os.path.join(root, filename)
                # Load and process the file
                loader = PyPDFLoader(file_path=file_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0
                )
                chunks = text_splitter.split_documents(data)

                cost_of_embedding_file = embedding_cost_calculator(chunks=chunks)
                total_estimated_cost += float(cost_of_embedding_file)

                print(cost_of_embedding_file)

    return total_estimated_cost

def sanitize_filename(filename):
    """Sanitize the filename to ensure it is ASCII-only."""
    # Replace non-ASCII characters with their closest ASCII equivalents
    sanitized = unidecode.unidecode(filename)
    # Optionally, remove or replace other non-ASCII characters or spaces as needed
    sanitized = sanitized.replace(" ", "_").replace("/", "_")
    return sanitized

def ingest_to_pinecone():
    for root, dirs, files in os.walk(docs_dir):
        for filename in files:
            if filename.endswith(".pdf"):  # Check if the file is a PDF
                try:
                    print("ingesting file:- " + filename)
                    file_path = os.path.join(root, filename)
                    # Load and process the file
                    loader = PyPDFLoader(file_path=file_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=0
                    )
                    chunks = text_splitter.split_documents(data)

                    for i, chunk in enumerate(chunks, start=1):
                        vector_id = sanitize_filename(f"{filename}_{i}")
                        try:
                            vector = embedding.embed_query(chunk.page_content)
                            metadata = {"text": chunk.page_content}
                            index.upsert(vectors=[(vector_id, vector, metadata)])

                        except Exception as e:
                            print(f"Error upserting chunk {i} of {filename}: {e}")
                except Exception as e:
                      print(f"Error processing file {filename}: {e}")



                            #Store in Pinecone



print("Total estimated cost is:- " + str(estimate_total_cost()))
if input("Would you like to continue? Press y for yes:- ") == "y":
    print("Starting ingest_to_pinecone")
    ingest_to_pinecone()
else:
    print("Operation Aborted!")


#Gammel Pinecone initializer fra inderen
"""pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV"),
)"""

# Gammel uploading...fra inderen
"""
Pinecone.from_documents(
    index_name=os.getenv("PINECONE_ENV"),
    embedding=embedding,
    documents=chunks,
) """
