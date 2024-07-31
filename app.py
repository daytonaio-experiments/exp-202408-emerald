import os
import ast
import dotenv
from openai import AzureOpenAI
from scipy.spatial.distance import cosine
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

import streamlit as st
from pprint import pprint

# Load environment variables from .env file
dotenv.load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)

# Initialize Qdrant client
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# Function to extract functions from a Python file
def extract_functions_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()

        tree = ast.parse(file_content)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_code = ast.get_source_segment(file_content, node)
                functions.append({
                    'name': node.name,
                    'code': function_code,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno
                })
        return functions
    except SyntaxError as e:
        st.error(f"Syntax error in file {file_path}: {e}")
        return []

# Function to preprocess the entire codebase
def preprocess_codebase(root_dir):
    code_chunks = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                functions = extract_functions_from_file(file_path)
                for func in functions:
                    chunk_metadata = {
                        'file_path': file_path,
                        'function_name': func['name'],
                        'start_line': func['start_line'],
                        'end_line': func['end_line'],
                        'code': func['code']
                    }
                    code_chunks.append(chunk_metadata)
    return code_chunks

# Function to create embeddings for code chunks
def create_embeddings(code_chunks):
    embeddings = []
    for chunk in code_chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk['code']
        )
        embedding = response.data[0].embedding
        embeddings.append({
            'file_path': chunk['file_path'],
            'function_name': chunk['function_name'],
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'code': chunk['code'],
            'embedding': embedding
        })
    return embeddings

# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(embeddings, project_name):
    # Create collection if it doesn't exist
    try:
        qdrant_client.get_collection(project_name)
    except Exception as e:
        qdrant_client.create_collection(
            collection_name=project_name,
            vectors_config=VectorParams(
                size=len(embeddings[0]['embedding']),  # Dimension of your vectors
                distance=Distance.COSINE
            )
        )
    
    points = []
    for i, embedding in enumerate(embeddings):
        points.append(PointStruct(
            id=i,
            vector=embedding['embedding'],
            payload={
                'file_path': embedding['file_path'],
                'function_name': embedding['function_name'],
                'start_line': embedding['start_line'],
                'end_line': embedding['end_line'],
                'code': embedding['code']
            }
        ))
    qdrant_client.upsert(collection_name=project_name, points=points)

# Function to look for the main.py file in the codebase and extract the main code
def extract_main_code(root_dir):
    main_file_path = os.path.join(root_dir, 'main.py')
    if os.path.exists(main_file_path):
        with open(main_file_path, 'r') as file:
            return file.read()
    return None

# Function to calculate cosine similarity
def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to query codebase and retrieve relevant code chunks
def query_codebase(query, project_name, top_n=5):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    search_results = qdrant_client.search(
        collection_name=project_name,
        query_vector=query_embedding,
        limit=top_n
    )

    results = [(result.score, result.payload) for result in search_results]
    return results

# Streamlit app
st.set_page_config(page_title="Daytona Experiments", layout="wide", page_icon="✨")
st.title("_⚡Intelligent Codebase Embeddings and Query System⚡_")

st.header("- *Flowchart*")
# col1, col2, col3 = st.columns([1.75, 5, 2.5])
# with col2:
st.image("flowchart.png", caption="flowchart.png")

# Initialize session state
if "code_chunks" not in st.session_state:
    st.session_state.code_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "main_code" not in st.session_state:
    st.session_state.main_code = None
if "results" not in st.session_state:
    st.session_state.results = []

# Preprocess codebase
st.header("- *Preprocess Codebase*")
root_directory = st.text_input("Enter the root directory of the codebase:", "./project-2")
project_name = st.text_input("Enter the project name:", "default_project")
if st.button("Preprocess Codebase"):
    st.session_state.code_chunks = preprocess_codebase(root_directory)
    st.write("Code chunks extracted:")
    st.write(st.session_state.code_chunks)

    # Create embeddings
    st.write("Creating embeddings for code chunks...")
    st.session_state.embeddings = create_embeddings(st.session_state.code_chunks)
    st.write("Embeddings created.")

    # Store embeddings in Qdrant
    st.write("Storing embeddings in Qdrant...")
    store_embeddings_in_qdrant(st.session_state.embeddings, project_name)
    st.write("Embeddings stored.")

    # Extract main code
    st.session_state.main_code = extract_main_code(root_directory)
    if st.session_state.main_code:
        st.write("\nMain code extracted from main.py:")
        st.code(st.session_state.main_code)

# Display code chunks and main code if they exist
if st.session_state.code_chunks:
    st.write("Code chunks extracted:")
    st.write(st.session_state.code_chunks)

if st.session_state.main_code:
    st.write("\nMain code extracted from main.py:")
    st.code(st.session_state.main_code)

# Query codebase
st.header("- *Query Codebase*")
query = st.text_input("Enter your query:")
if st.button("Query Codebase"):
    st.session_state.results = query_codebase(query, project_name)
    st.write("Top results:")
    for score, doc in st.session_state.results:
        st.write(f"Score: {score}")
        st.write(f"File Path: {doc['file_path']}")
        st.write(f"Function Name: {doc['function_name']}")
        st.code(doc['code'])
        st.write("-" * 40)
