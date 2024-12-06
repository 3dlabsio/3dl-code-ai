import streamlit as st
import deeplake
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import argparse

def load_database(deeplake_path):
    """Load the Deep Lake dataset"""
    return deeplake.open(deeplake_path)

def get_relevant_chunks(query, dataset, k=5):
    """Search Deep Lake for relevant chunks using OpenAI embeddings"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get query embedding
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Simple cosine similarity search with proper syntax
    query_str = f"""
    SELECT text_chunk
    FROM (
        SELECT *, cosine_similarity(embedding, ARRAY[{','.join(map(str, query_embedding))}]) AS similarity_score 
    )
    ORDER BY similarity_score DESC
    LIMIT {k}
    """
    
    results = dataset.query(query_str)
    return [result['text_chunk'] for result in results]

def get_chatbot_response(query, context_chunks):
    """Get response from ChatGPT using context"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """You are a helpful assistant answering questions about a code repository. 
    Use the provided context to answer questions. If you're not sure about something, say so."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_chunks}\n\nQuestion: {query}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def main():
    dataset_path = st.sidebar.text_input(
        "Deep Lake Dataset Path", 
        value="hub://shanewarner/shippo-python-client",
        help="Format: hub://<username>/<dataset-name>"
    )
    ds = load_database(dataset_path)
    load_dotenv()
    
    st.title("Code Repository Chat Assistant")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load Deep Lake dataset
    dataset_path = "hub://shanewarner/shippo-python-client"  # Your dataset path
    ds = load_database(dataset_path)
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about the code repository..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get relevant chunks from Deep Lake
        with st.spinner("Searching repository..."):
            relevant_chunks = get_relevant_chunks(prompt, ds)
        
        # Get chatbot response
        with st.spinner("Thinking..."):
            response = get_chatbot_response(prompt, relevant_chunks)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", 
                       default="hub://shanewarner/shippo-python-client",
                       help="Deep Lake dataset path (e.g., hub://username/dataset)")
    args = parser.parse_args()
    
    # Configure Streamlit
    st.set_page_config(page_title="Code Repository Chat Assistant")
    streamlit_config = {
        'server.address': '0.0.0.0',
        'server.port': 8501
    }
    for key, value in streamlit_config.items():
        st.config.set_option(key, value)
    
    # Set the dataset path in session state
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = args.dataset_path
    
    main()