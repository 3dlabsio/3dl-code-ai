import os
import sys
import shutil
import git
from dotenv import load_dotenv
from urllib.parse import urlparse
import deeplake
import openai
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clone_repository(repo_url, clone_dir="./cloned_repo"):
    """Clones the repository from the given URL to a local directory."""
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    print(f"Cloning repository from {repo_url} into {clone_dir}...")
    git.Repo.clone_from(repo_url, clone_dir)
    print("Repository cloned successfully.")
    return clone_dir

def load_documents(directory):
    """Walks through the repository directory and loads documents."""
    print("Loading documents from the repository...")
    docs = []
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            try:
                filepath = os.path.join(dirpath, file)
                # Try multiple encodings
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings_to_try:
                    try:
                        with open(filepath, 'r', encoding=encoding) as f:
                            docs.append(f.read())
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"Skipping {file}: {e}")
    print(f"Loaded {len(docs)} documents.")
    return docs

def chunk_documents(docs, chunk_size=1000):
    """Splits documents into smaller chunks."""
    print(f"Splitting documents into chunks of size {chunk_size}...")
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i:i + chunk_size])
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def embedding_function(texts, model="text-embedding-ada-002"):
    """Embeds the texts using OpenAI's updated client API."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(model=model, input=texts)
    embeddings = [item.embedding for item in response.data]
    return embeddings


def store_in_deeplake(chunks, deeplake_dataset_path):
    """Stores the processed chunks in a Deep Lake dataset."""
    print(f"Storing chunks in Deep Lake dataset: {deeplake_dataset_path}...")

    # Check if the dataset exists and delete it if so
    if deeplake.exists(deeplake_dataset_path):
        print(f"Dataset already exists at {deeplake_dataset_path}. Deleting it...")
        deeplake.delete(deeplake_dataset_path)

    # Create the dataset
    ds = deeplake.dataset(deeplake_dataset_path)
    
    # Prepare embeddings
    embeddings = embedding_function(chunks)

    # Add data to the dataset
    with ds:
        ds.text.extend(chunks)
        ds.embedding.extend(embeddings)

    print("Documents stored successfully.")


def main():
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
    
    if not OPENAI_API_KEY or not ACTIVELOOP_TOKEN:
        print("Error: Missing OpenAI or ActiveLoop API credentials in .env file.")
        sys.exit(1)
    
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python codeai.py <repository_url>")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    repo_name = os.path.splitext(os.path.basename(urlparse(repo_url).path))[0]
    clone_dir = "./cloned_repo"
    deeplake_dataset_path = f"al://{os.getenv('DEEPLAKE_USERNAME', 'default_user')}/{repo_name}"
    
    # Process repository
    cloned_repo_path = clone_repository(repo_url, clone_dir)
    documents = load_documents(cloned_repo_path)
    chunks = chunk_documents(documents)
    store_in_deeplake(chunks, deeplake_dataset_path)

    print("Processing complete. The repository has been indexed in Deep Lake.")

if __name__ == "__main__":
    main()
