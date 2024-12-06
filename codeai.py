import os
import sys
import time
import shutil
import git
from dotenv import load_dotenv
from urllib.parse import urlparse
import deeplake
from openai import OpenAI
import numpy as np

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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if isinstance(texts, str):
        texts = [texts]
        
    # Preprocess texts:
    # 1. Convert to string
    # 2. Replace newlines with spaces
    # 3. Handle empty strings
    processed_texts = []
    for text in texts:
        if text is None or text.strip() == "":
            text = " "  # OpenAI API doesn't accept empty strings
        processed_text = str(text).replace("\n", " ").strip()
        processed_texts.append(processed_text)
    
    # Check if any text is empty after processing
    if not any(processed_texts):
        raise ValueError("All texts are empty after processing")
        
    # Create embeddings in smaller batches to avoid token limits
    batch_size = 100  # Adjust based on your needs
    all_embeddings = []
    
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=model,
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            raise
            
    return all_embeddings

def store_in_deeplake(chunks, deeplake_dataset_path):
    """Stores the processed chunks in a Deep Lake dataset."""
    print(f"Storing chunks in Deep Lake dataset: {deeplake_dataset_path}...")

    # Try to delete existing dataset
    try:
        deeplake.delete(deeplake_dataset_path)
        print(f"Existing dataset at {deeplake_dataset_path} deleted.")
    except Exception as e:
        print(f"No existing dataset found to delete: {e}")

    # Create a dataset with the TextEmbeddings schema
    ds = deeplake.create(
        deeplake_dataset_path,
        schema=deeplake.schemas.TextEmbeddings(embedding_size=1536)
    )

    # Generate embeddings for all chunks
    print("Generating embeddings for all chunks...")
    embedding_vectors = embedding_function(chunks)

    # Prepare and add the data
    print("Populating dataset...")
    
    # Create dictionary with lists for batch append
    current_time = int(time.time())
    ds.append({
        "id": list(range(len(chunks))),
        "date_created": [current_time] * len(chunks),
        "document_id": list(range(len(chunks))),
        "document_url": [""] * len(chunks),
        "text_chunk": chunks,
        "license": [""] * len(chunks),
        "embedding": embedding_vectors
    })

    # Commit the changes
    ds.commit()
    print("Documents stored successfully.")
    
    # Print summary
    ds.summary()


def verify_embeddings(deeplake_dataset_path):
    ds = deeplake.open(deeplake_dataset_path)
    
    # Get first row
    first_row = ds[0]
    
    # Check first embedding and text
    first_embedding = first_row['embedding']
    first_text = first_row['text_chunk']
    
    print("\nFirst text chunk:", first_text[:100], "...")  # Show first 100 chars
    print(f"\nEmbedding verification:")
    print(f"Shape: {first_embedding.shape}")
    print(f"Sample values: {first_embedding[:5]}")
    print(f"L2 norm: {np.linalg.norm(first_embedding):.6f}")  # Should be ~1.0
    print(f"Min/Max: {first_embedding.min():.6f} / {first_embedding.max():.6f}")

    # Also show a few more rows to verify consistency
    print("\nVerifying multiple rows:")
    for row in ds[:3]:  # Check first 3 rows
        emb = row['embedding']
        print(f"Embedding shape: {emb.shape}")


def main():
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
    
    if not OPENAI_API_KEY or not ACTIVELOOP_TOKEN:
        print("Error: Missing OpenAI or ActiveLoop API credentials in .env file.")
        sys.exit(1)
    
    # Set environment variables for both APIs
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python codeai.py <repository_url>")
        sys.exit(1)
    
    repo_url = sys.argv[1]
    repo_name = os.path.splitext(os.path.basename(urlparse(repo_url).path))[0]
    clone_dir = "./cloned_repo"
    deeplake_dataset_path = f"hub://{os.getenv('DEEPLAKE_USERNAME', 'default_user')}/{repo_name}"
    
    # Process repository
    cloned_repo_path = clone_repository(repo_url, clone_dir)
    documents = load_documents(cloned_repo_path)
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("No documents were loaded. Exiting.")
        sys.exit(1)
        
    store_in_deeplake(chunks, deeplake_dataset_path)
    print("Processing complete. The repository has been indexed in Deep Lake.")

    # Add verification after storing
    verify_embeddings(deeplake_dataset_path)

if __name__ == "__main__":
    main()