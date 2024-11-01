from agent import Agent
import streamlit as st
import tempfile

def create_app():
    # Streamlit App layout
    st.title("Smart AI Chat Assistant")
    # st.subheader("Chat with the bot")

    # Option to choose between chat mode or upload file mode
    mode = st.radio("Choose Mode", ("Chat with Bot", "Upload File"))

   

    # Initialize PDF mode toggle and retriever state
    if "pdf_mode" not in st.session_state:
        st.session_state.pdf_mode = False
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # Initialize the Agent object only once
    if "agent1" not in st.session_state and "agent2" not in st.session_state:
        st.session_state.agent1 = Agent()
        st.session_state.agent2 = Agent()

    # If mode is "Chat with Bot" (default mode)
    if mode == "Chat with Bot":
        user_input = st.text_input("You: ")
        if user_input:
           
            
            # Use the already initialized agent from session state
            bot_response = st.session_state.agent1.search_agent(query=user_input, tool_type='search')
          

        # Display chat history in reverse order
        for message in st.session_state.agent1.memory.dict()['chat_memory']['messages'][::-1]:
            if message["type"] == "human":
                st.write(f"**You**: {message['content']}")
            elif message["type"] == "ai":
                st.write(f"**Bot**: {message['content']}")

    # If mode is "Upload File" (PDF mode)
    elif mode == "Upload File":
        st.subheader("Upload a File for Analysis")

        # Upload file functionality
        uploaded_file = st.file_uploader("Choose a text file", type="txt")

        if st.button("Upload and Process"):
            if uploaded_file is not None:
                # Set PDF mode to True and save the retriever in session state
                st.session_state.pdf_mode = True

                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_file_path = tmp_file.name

                # Initialize the retriever in the Agent
                st.session_state.retriever = st.session_state.agent1.pdf_agent(query_file=temp_file_path)
                st.write("File uploaded successfully! You can now start chatting based on the file content.")

        # Only show chat interface after a file has been uploaded and processed
        if st.session_state.retriever:
            user_input = st.text_input("You: ")
            if user_input:
             
                bot_response = st.session_state.agent2.search_agent(query=user_input, tool_type='doc_retr', retriever=st.session_state.retriever)
               

            # Display chat history in reverse order
            for message in st.session_state.agent2.memory.dict()['chat_memory']['messages'][::-1]:
                if message["type"] == "human":
                    st.write(f"**You**: {message['content']}")
                elif message["type"] == "ai":
                    st.write(f"**Bot**: {message['content']}")

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Chat-AI")
    create_app()
