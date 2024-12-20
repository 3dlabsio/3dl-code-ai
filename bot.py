import streamlit as st
import deeplake
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import argparse
import uuid
from db_helper import DatabaseHelper

# Initialize database helper
db = DatabaseHelper()

def get_default_dataset_path():
    """Get the default dataset path using environment variables"""
    username = os.getenv('DEEPLAKE_USERNAME', '')
    if not username:
        return "hub://<username>/"
    return f"hub://{username}/"

def load_database(deeplake_path):
    """Load the Deep Lake dataset"""
    return deeplake.open(deeplake_path)

def get_relevant_chunks(query, dataset, k=2):  # Reduced from 3 to 2
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
    chunks = [result['text_chunk'] for result in results]
    
    # Debug printing
    print("\n📚 Retrieved chunks for context:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print("=" * 40)
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print("=" * 40)
    
    return chunks

def get_chatbot_response(query, context_chunks, session_id):
    """Get response from ChatGPT using context and chat history"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get recent chat history - reduced from 5 to 3
    history = db.get_session_history(session_id, limit=3)
    
    # Get recent context history - reduced from 3 to 2
    context_history = db.get_context_history(session_id, limit=2)
    
    system_prompt = """You are an expert programmer that can analyze code in a repository and provide helpful, contextual responses. 
    Use the provided context and chat history to answer questions.
    
    Important instructions:
    1. Always reference which specific parts of the context you used in your answer
    2. If you can't find relevant information in the context, say so explicitly
    3. If you're making assumptions beyond the context, indicate this clearly
    4. Quote relevant code snippets when they support your answer
    5. Consider the chat history for better context
    
    Format your responses with clear sections:
    - Answer: Your main response
    - Sources Used: List the relevant chunks you used, with brief explanations
    - Confidence: High/Medium/Low, with explanation"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_chunks}\n\nPrevious Context:\n{context_history}\n\nQuestion: {query}"}
    ]
    
    # Add chat history
    for role, content, _ in history:
        messages.append({"role": role, "content": content})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def init_session_state():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.dataset_path = get_default_dataset_path()
        st.session_state.need_load = True
        st.session_state.current_loaded_dataset = None  # Track currently loaded dataset
        # Create initial session in database
        db.create_session(st.session_state.session_id, st.session_state.dataset_path)
    else:
        # Load current session info if session_id exists
        if hasattr(st.session_state, 'session_id'):
            session_info = db.get_session_info(st.session_state.session_id)
            if session_info:
                st.session_state.dataset_path = session_info['dataset_path']

def load_session(session_id, dataset_path):
    """Load a specific session and its data"""
    print(f"Loading session {session_id} with dataset {dataset_path}")
    st.session_state.session_id = session_id
    st.session_state.dataset_path = dataset_path
    # Only set need_load if the dataset is different from currently loaded one
    st.session_state.need_load = (dataset_path != st.session_state.get('current_loaded_dataset'))
    if st.session_state.need_load:
        print(f"Dataset {dataset_path} differs from current {st.session_state.get('current_loaded_dataset')}, will reload")
    else:
        print(f"Dataset {dataset_path} already loaded, skipping reload")

def display_chat_history(session_id):
    """Display chat history with proper error handling"""
    try:
        history = list(reversed(db.get_session_history(st.session_state.session_id)))
        print(f"\nDebug: Found {len(history)} messages in history")
        
        for i, (role, content, _) in enumerate(history):
            print(f"\nMessage {i + 1}:")
            print(f"Role: {repr(role)}")
            print(f"Content type: {type(content)}")
            print(f"Content: {repr(content)[:100]}...")
            
            try:
                # Ensure role is string and one of expected values
                if not isinstance(role, str) or role not in ['user', 'assistant']:
                    print(f"Skipping message {i + 1} due to invalid role: {role}")
                    continue
                
                # Ensure content is string and non-empty
                if not isinstance(content, str) or not content.strip():
                    print(f"Skipping message {i + 1} due to invalid content")
                    continue
                
                with st.chat_message(role):
                    st.markdown(content)
            except Exception as e:
                print(f"Error displaying message {i + 1}: {str(e)}")
                
    except Exception as e:
        print(f"Error loading chat history: {str(e)}")
        st.error("Failed to load chat history. Check the logs for details.")

def display_chat_list():
    """Display and manage chat sessions in the sidebar"""
    st.sidebar.title("Chat Sessions")
    
    # Get all sessions first
    sessions = db.get_all_sessions()
    
    # Store sessions in session state for reference
    if "current_sessions" not in st.session_state:
        st.session_state.current_sessions = sessions
    
    # New chat button
    if st.sidebar.button("New Chat", key="new_chat"):
        new_session_id = str(uuid.uuid4())
        default_path = get_default_dataset_path()
        db.create_session(new_session_id, default_path)
        load_session(new_session_id, default_path)
        st.rerun()

    # Display sessions
    for session in sessions:
        # Highlight current session with an emoji indicator
        is_current = session['session_id'] == st.session_state.get('session_id')
        session_title = f"{'→ ' if is_current else ''}{session['title'] or 'Untitled Chat'}"
        
        col1, col2 = st.sidebar.columns([4, 1])
        
        # Session title/button
        with col1:
            if st.button(
                session_title,
                key=f"session_{session['session_id']}",
                use_container_width=True
            ):
                load_session(session['session_id'], session['dataset_path'])
                st.rerun()
        
        # Delete button
        with col2:
            if st.button("🗑️", key=f"delete_{session['session_id']}"):
                db.delete_session(session['session_id'])
                if session['session_id'] == st.session_state.session_id:
                    # Find another session to load
                    other_session = next((s for s in sessions if s['session_id'] != session['session_id']), None)
                    if other_session:
                        load_session(other_session['session_id'], other_session['dataset_path'])
                    else:
                        # Create new session if no others exist
                        new_session_id = str(uuid.uuid4())
                        default_path = get_default_dataset_path()
                        db.create_session(new_session_id, default_path)
                        load_session(new_session_id, default_path)
                st.rerun()

def main():
    load_dotenv()
    
    # Initialize session state
    init_session_state()
    
    # Display chat list in sidebar
    display_chat_list()
    
    # Add a separator in sidebar
    st.sidebar.markdown("---")
    
    # Dataset path in sidebar with current session's path
    dataset_path = st.sidebar.text_input(
        "Deep Lake Dataset Path", 
        value=st.session_state.dataset_path,
        key="dataset_input",
        help="Format: hub://<username>/<repository-name>"
    )
    
    # Handle dataset path changes
    if dataset_path != st.session_state.dataset_path:
        st.session_state.dataset_path = dataset_path
        st.session_state.need_load = True
        # Update session info in database
        repo_name = dataset_path.split('/')[-1] or "New Chat"
        db.update_session_title(st.session_state.session_id, f"Chat: {repo_name}")
        db.update_session_dataset(st.session_state.session_id, dataset_path)
        st.rerun()
    
    st.title("Code Repository Chat Assistant")
    
    # Debug information
    if True:  # Changed from os.getenv('DEBUG') to always show during troubleshooting
        st.sidebar.markdown("---")
        st.sidebar.write("Debug Info:")
        st.sidebar.write(f"Session ID: {st.session_state.session_id}")
        st.sidebar.write(f"Dataset Path: {dataset_path}")
        st.sidebar.write(f"Need Load: {st.session_state.get('need_load', True)}")
        st.sidebar.write(f"Current Loaded Dataset: {st.session_state.get('current_loaded_dataset')}")
    
    # Check if a repository is selected
    if dataset_path == get_default_dataset_path():
        st.info("Please enter a specific repository path in the sidebar to begin chatting.")
        return

    # Load dataset if needed
    ds = None
    if st.session_state.get('need_load', True):
        st.write("Loading dataset...")
        try:
            ds = load_database(dataset_path)
            st.session_state.need_load = False
            st.session_state.current_loaded_dataset = dataset_path  # Update currently loaded dataset
            print(f"Successfully loaded dataset: {dataset_path}")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
    else:
        try:
            ds = load_database(dataset_path)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
    
    # Display chat history
    print("\nAttempting to display chat history...")
    display_chat_history(st.session_state.session_id)
    
    if prompt := st.chat_input("Ask about the code repository..."):
        print(f"\nNew message received - Role: user, Content: {repr(prompt)}")
        
        # Add user message to chat history
        db.add_message(st.session_state.session_id, "user", prompt)
        
        try:
            with st.chat_message("user"):
                st.markdown(prompt)
        except Exception as e:
            print(f"Error displaying user message: {str(e)}")
            
        # Get relevant chunks from Deep Lake
        with st.spinner("Searching repository..."):
            try:
                relevant_chunks = get_relevant_chunks(prompt, ds)
            except Exception as e:
                st.error(f"Error searching repository: {str(e)}")
                return
                
        # Get chatbot response
        with st.spinner("Thinking..."):
            try:
                response = get_chatbot_response(prompt, relevant_chunks, st.session_state.session_id)
                # Add and display assistant response
                db.add_message(st.session_state.session_id, "assistant", response, relevant_chunks)
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(page_title="Code Repository Chat Assistant")
    main()