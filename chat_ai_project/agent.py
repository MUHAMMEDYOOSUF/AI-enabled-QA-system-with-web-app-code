from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent,AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader,PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from pathlib import Path
import os


BASE=Path(__file__).parent.absolute()


class VectorDB:
    def __init__(self,collection_name):
        # Initialize the embedding model and Chroma vector store
        self.huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.collection_name=collection_name
        self.chroma_vector_store = Chroma(
            embedding_function=self.huggingface_embeddings,
            persist_directory='./new_db',
            collection_name=self.collection_name
        )
    

    def add_data_to_collection(self,documents):
        self.chroma_vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to collection: {self.collection_name}")





class Agent:

    def __init__(self):
        self.groq_api_key='gsk_4fmsu1kuesOh9yQe1xEerwJhkasiL78Lwu8nKvULBp'
        self.google_api_key='AIzarqererqerqrqetwtdgdggoY'
        self.search_engine_id='43arrsrf56785756889'
        self.llm=ChatGroq(model='llama-3.1-70b-versatile',api_key=self.groq_api_key,temperature=1e-08)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def search_agent(self,query,tool_type,retriever=None):
        tools=[]
        if tool_type=='search':
            api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            tool_wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
            search = GoogleSearchAPIWrapper(google_api_key=self.google_api_key,google_cse_id=self.search_engine_id)
            tool_google = Tool(
                name="google_search",
                description="Search Google for recent results.",
                func=search.run,
            )
            tools=[
                tool_google,tool_wiki
            ]
        if tool_type=='doc_retr':
            doc_tool = create_retriever_tool(
                retriever,
                "blog_post_retriever",
                "Searches and returns excerpts from the Autonomous Agents blog post.",
            )
            tools=[doc_tool]
        prompt=PromptTemplate(
            input_variables=['chat_history', 'agent_scratchpad', 'input', 'tool_names', 'tools'],
            metadata={
                'lc_hub_owner': 'hwchase17',
                'lc_hub_repo': 'react',
                'lc_hub_commit_hash': 'd15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d'
            },
            template=(
                'You are an intelligent assistant that can answer questions and perform actions using the tools provided.\n'
                'You also have access to the conversation history, which you should use to provide more relevant and context-aware responses.\n\n'
                'Here are the available tools:\n\n{tools}\n\n'
                'Here is the chat history so far:\n\n{chat_history}\n\n'
                'When responding, take the chat history into account.\n'
                'Use the following format to answer the questions:\n\n'
                'Question: the input question you must answer\n'
                'Thought: you should always think about what to do based on the chat history and tools\n'
                'Action: the action to take, should be one of [{tool_names}]\n'
                'Action Input: the input to the action\n'
                'Observation: the result of the action\n'
                '... (this Thought/Action/Action Input/Observation can repeat N times)\n'
                'Thought: I now know the final answer\n'
                'Final Answer: the final answer to the original input question, considering the chat history\n\n'
                'Begin!\n\n'
                'Question: {input}\n'
                'Thought: {agent_scratchpad}'
            )
        )
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
            stop_sequence=True,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools,memory=self.memory,handle_parsing_errors=True)
        try:
            response=agent_executor.invoke({"input": query})
        except ValueError as e:
            print(f"Error processing request: {e}")
            return "There was an error processing your request. Please try again."
        return response['output']
    


    def pdf_agent(self,query_file):
      
        vector_db = VectorDB(collection_name='col1')
        if vector_db.chroma_vector_store._client.get_collection('col1').count()!=0:
            print("Data exists")
            vector_db.chroma_vector_store._client.delete_collection('col1')
            print("Deleted collection")
            vector_db = VectorDB(collection_name='col1')
            print("Created collection")

        loader=TextLoader(query_file)
        documents=loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
    
        vector_db.add_data_to_collection(docs)
        chroma_retreiver=vector_db.chroma_vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.1},
        )
        return chroma_retreiver






