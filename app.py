import os
import ast
import dotenv
import subprocess
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

# Function to get the directory structure
def get_tree_structure(path):
    # Run the tree command with /F flag using cmd.exe (Windows specific)
    result = subprocess.run(['cmd.exe', '/c', f'tree /F {path}'], capture_output=True, text=True)
    # Store the output in a variable
    tree_output = result.stdout
    return tree_output

# Function to generate a description for a code chunk
def generate_description(code):
    prompt = f"Summarize the purpose of the following Python code in at least 2 sentences:\n\n{code}\n\nSummary:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating description: {e}")
        return "No description available"

# Function to extract functions and classes from a Python file
def extract_functions_and_classes_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()

        tree = ast.parse(file_content)
        entities = []  # To store both functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_code = ast.get_source_segment(file_content, node)
                entities.append({
                    'type': 'function',
                    'name': node.name,
                    'code': function_code,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno
                })
            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(file_content, node)
                entities.append({
                    'type': 'class',
                    'name': node.name,
                    'code': class_code,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno
                })
        return entities
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
                entities = extract_functions_and_classes_from_file(file_path)
                for entity in entities:
                    chunk_metadata = {
                        'file_path': file_path,
                        'entity_type': entity['type'],
                        'entity_name': entity['name'],
                        'start_line': entity['start_line'],
                        'end_line': entity['end_line'],
                        'code': entity['code']
                    }
                    code_chunks.append(chunk_metadata)
    return code_chunks

# Function to create embeddings for code chunks
def create_embeddings(code_chunks):
    embeddings = []
    for chunk in code_chunks:
        # Generate a description for the code chunk
        description = generate_description(chunk['code'])
        
        # Create the input text for embedding that includes all relevant information
        input_text = (
            f"File Path: {chunk['file_path']}\n"
            f"Entity Type: {chunk['entity_type']}\n"
            f"Entity Name: {chunk['entity_name']}\n"
            f"Start Line: {chunk['start_line']}\n"
            f"End Line: {chunk['end_line']}\n"
            f"Description: {description}\n"
            f"Code:\n{chunk['code']}"
        )
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=input_text
        )
        embedding = response.data[0].embedding
        
        # Append the chunk with the embedding and description
        embeddings.append({
            'file_path': chunk['file_path'],
            'entity_type': chunk['entity_type'],
            'entity_name': chunk['entity_name'],
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'code': chunk['code'],
            'embedding': embedding,
            'isFunction': chunk['entity_type'] == 'function',
            'description': description
        })
    return embeddings

# Function to store embeddings in Qdrant
def store_embeddings_in_qdrant(embeddings, project_name, root_dir):
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
    
    # Get the directory structure of the root directory
    directory_structure = get_tree_structure(root_dir)
    
    points = []
    for i, embedding in enumerate(embeddings):
        points.append(PointStruct(
            id=i,
            vector=embedding['embedding'],
            payload={
                'file_path': embedding['file_path'],
                'entity_type': embedding['entity_type'],
                'entity_name': embedding['entity_name'],
                'start_line': embedding['start_line'],
                'end_line': embedding['end_line'],
                'code': embedding['code'],
                'description': embedding['description'],
                'isFunction': embedding['isFunction'],
                'directory_structure': directory_structure  # Store the directory structure
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
root_directory = st.text_input("Enter the root directory of the codebase:", "./project")
project_name = st.text_input("Enter the project name:", "project")
if st.button("Preprocess Codebase"):
    st.session_state.code_chunks = preprocess_codebase(root_directory)
    st.write("Code chunks extracted:")
    st.write(st.session_state.code_chunks)

    # Display the project directory structure
    st.write("Project Directory:")
    st.code(get_tree_structure(root_directory))

    # Create embeddings
    with st.spinner("Creating embeddings for code chunks..."):
        st.session_state.embeddings = create_embeddings(st.session_state.code_chunks)
    st.toast("Embeddings created", icon="🚀")

    # Store embeddings in Qdrant
    with st.spinner("Storing embeddings in Qdrant..."):
        store_embeddings_in_qdrant(st.session_state.embeddings, project_name, root_directory)
    st.toast("Embeddings stored", icon="🚀")

    st.success("Codebase preprocessing complete.")

    # Extract main code
    st.session_state.main_code = extract_main_code(root_directory)
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
        st.write(f"Function Name: `{doc['entity_name']}()`")
        st.code(doc['code'])
        st.write("-" * 40)
