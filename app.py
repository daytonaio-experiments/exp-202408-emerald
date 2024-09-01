import os
import ast
from dotenv import load_dotenv
import requests
from openai import AzureOpenAI
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from pprint import pprint
import logging
from tqdm import tqdm
import time
import backoff

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Read the API key from the environment variable
azure_api_version = os.getenv("AZURE_API_VERSION")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="db")

# Initialize Azure OpenAI Embedding Function
azure_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=azure_api_key,
    api_base=azure_endpoint,
    api_type="azure",
    api_version=azure_api_version,
    model_name="text-embedding-3-small",
)

# GitHub API related functions
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def get_github_tree(repo_owner, repo_name, github_token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/main?recursive=1"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def get_file_content(repo_owner, repo_name, file_path, github_token):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = response.json()["content"]
    import base64
    return base64.b64decode(content).decode('utf-8')

# Function to get the directory structure
def get_tree_structure(repo_owner, repo_name, github_token):
    logger.info(f"Fetching directory structure for {repo_owner}/{repo_name}")
    tree = get_github_tree(repo_owner, repo_name, github_token)
    structure = []
    for item in tree['tree']:
        structure.append(item['path'])
    return "\n".join(structure)

# Function to generate a description for a code chunk
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_description(code):
    prompt = f"Summarize the purpose of the following Python code in at least 2 sentences:\n\n{code}\n\nSummary:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        return "No description available"

# Function to extract functions and classes from a Python file
def extract_functions_and_classes_from_content(file_content, file_path):
    try:
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
        logger.error(f"Syntax error in file {file_path}: {e}")
        return []

# Function to preprocess the entire codebase
def preprocess_codebase(repo_owner, repo_name, github_token):
    logger.info(f"Starting codebase preprocessing for {repo_owner}/{repo_name}")
    code_chunks = []
    tree = get_github_tree(repo_owner, repo_name, github_token)
    for item in tqdm(tree['tree'], desc="Processing files"):
        if item['path'].endswith('.py'):
            try:
                file_content = get_file_content(repo_owner, repo_name, item['path'], github_token)
                entities = extract_functions_and_classes_from_content(file_content, item['path'])
                for entity in entities:
                    chunk_metadata = {
                        'file_path': item['path'],
                        'entity_type': entity['type'],
                        'entity_name': entity['name'],
                        'start_line': entity['start_line'],
                        'end_line': entity['end_line'],
                        'code': entity['code']
                    }
                    code_chunks.append(chunk_metadata)
            except Exception as e:
                logger.error(f"Error processing file {item['path']}: {e}")
    logger.info(f"Codebase preprocessing completed. Extracted {len(code_chunks)} code chunks.")
    return code_chunks

# Function to create embeddings for code chunks
def create_embeddings(code_chunks):
    logger.info("Starting embedding creation")
    embeddings = []
    for chunk in tqdm(code_chunks, desc="Creating embeddings"):
        try:
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

            embedding = azure_ef([input_text])[0]  # Note the change here: we pass a list

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
        except Exception as e:
            logger.error(f"Error creating embedding for chunk {chunk['entity_name']}: {e}")
    logger.info(f"Embedding creation completed. Created {len(embeddings)} embeddings.")
    return embeddings

# Function to store embeddings in ChromaDB
def store_embeddings_in_chromadb(embeddings, project_name, repo_owner, repo_name, github_token):
    logger.info(f"Storing embeddings in ChromaDB for project {project_name}")
    # Create or get collection
    collection = chroma_client.get_or_create_collection(name=project_name, embedding_function=azure_ef)

    # Get the directory structure of the repository
    directory_structure = get_tree_structure(repo_owner, repo_name, github_token)

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []
    embeddings_list = []

    for i, embedding in enumerate(tqdm(embeddings, desc="Preparing data for ChromaDB")):
        ids.append(str(i))
        documents.append(embedding['code'])
        metadatas.append({
            'file_path': embedding['file_path'],
            'entity_type': embedding['entity_type'],
            'entity_name': embedding['entity_name'],
            'start_line': embedding['start_line'],
            'end_line': embedding['end_line'],
            'description': embedding['description'],
            'isFunction': embedding['isFunction'],
            'directory_structure': directory_structure
        })
        embeddings_list.append(embedding['embedding'])

    # Add data to ChromaDB
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        logger.info(f"Successfully stored {len(embeddings)} embeddings in ChromaDB")
    except Exception as e:
        logger.error(f"Error storing embeddings in ChromaDB: {e}")

# Function to look for main files in the codebase and extract their code
def extract_main_files(repo_owner, repo_name, github_token):
    logger.info("Attempting to extract main files")
    main_files = ['main.py', 'script.py', 'app.py']
    extracted_files = {}
    for file in main_files:
        try:
            content = get_file_content(repo_owner, repo_name, file, github_token)
            extracted_files[file] = content
            logger.info(f"Successfully extracted {file}")
        except Exception as e:
            logger.warning(f"Could not extract {file}: {e}")
    return extracted_files

# Function to query codebase and retrieve relevant code chunks
def query_codebase(query, project_name, top_n=5):
    logger.info(f"Querying codebase with: '{query}'")
    collection = chroma_client.get_collection(name=project_name, embedding_function=azure_ef)
    results = collection.query(
        query_texts=[query],
        n_results=top_n
    )

    # Format results to match the previous structure
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append((
            results['distances'][0][i],  # Score
            {
                'file_path': results['metadatas'][0][i]['file_path'],
                'entity_name': results['metadatas'][0][i]['entity_name'],
                'code': results['documents'][0][i]
            }
        ))

    logger.info(f"Query returned {len(formatted_results)} results")
    return formatted_results

# Streamlit app
st.set_page_config(page_title="Daytona Experiments", layout="wide", page_icon="✨")
st.title("_⚡Intelligent Codebase Embeddings and Query System⚡_")

st.header("- *Flowchart*")
st.image("flowchart.png", caption="flowchart.png")

# Initialize session state
if "code_chunks" not in st.session_state:
    st.session_state.code_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "main_files" not in st.session_state:
    st.session_state.main_files = {}
if "results" not in st.session_state:
    st.session_state.results = []

# Preprocess codebase
st.header("- *Preprocess Codebase*")
github_repo_url = st.text_input("Enter the GitHub repository URL:", "https://github.com/owner/repo")
project_name = st.text_input("Enter the project name:", "project")
github_token = os.getenv("GITHUB_TOKEN")

if st.button("Preprocess Codebase"):
    try:
        # Extract owner and repo name from the URL
        parts = github_repo_url.split('/')
        repo_owner, repo_name = parts[-2], parts[-1]

        with st.spinner("Preprocessing codebase..."):
            st.session_state.code_chunks = preprocess_codebase(repo_owner, repo_name, github_token)
        st.success(f"Extracted {len(st.session_state.code_chunks)} code chunks")

        # Display the project directory structure
        st.write("Project Directory:")
        st.code(get_tree_structure(repo_owner, repo_name, github_token))

        # Display code chunks
        st.write("Code Chunks:")
        for chunk in st.session_state.code_chunks:
            st.write(f"File: {chunk['file_path']}, Entity: {chunk['entity_name']}")
            st.code(chunk['code'])
            st.write("---")

        # Create embeddings
        with st.spinner("Creating embeddings for code chunks..."):
            st.session_state.embeddings = create_embeddings(st.session_state.code_chunks)
        st.success(f"Created {len(st.session_state.embeddings)} embeddings")

        # Store embeddings in ChromaDB
        with st.spinner("Storing embeddings in ChromaDB..."):
            store_embeddings_in_chromadb(st.session_state.embeddings, project_name, repo_owner, repo_name, github_token)
        st.success("Embeddings stored in ChromaDB")

        # Extract main files
        st.session_state.main_files = extract_main_files(repo_owner, repo_name, github_token)
        if st.session_state.main_files:
            st.write("\nMain files extracted:")
            for file, content in st.session_state.main_files.items():
                st.write(f"File: {file}")
                st.code(content)
                st.write("---")
        else:
            st.warning("No main files (main.py, script.py, app.py) found in the repository")

        st.success("Codebase preprocessing complete.")
    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        logger.error(f"Preprocessing error: {e}", exc_info=True)

# Query codebase
st.header("- *Query Codebase*")
query = st.text_input("Enter your query:")
if st.button("Query Codebase"):
    try:
        with st.spinner("Querying codebase..."):
            st.session_state.results = query_codebase(query, project_name)
        st.write("Top results:")
        for score, doc in st.session_state.results:
            st.write(f"Score: {score}")
            st.write(f"File Path: {doc['file_path']}")
            st.write(f"Function Name: `{doc['entity_name']}()`")
            st.code(doc['code'])
            st.write("-" * 40)
    except Exception as e:
        st.error(f"An error occurred during querying: {e}")
        logger.error(f"Querying error: {e}", exc_info=True)

# Add a section to display and navigate through code chunks
st.header("- *Code Chunks*")
if st.session_state.code_chunks:
    chunk_index = st.selectbox("Select a code chunk:",
                               range(len(st.session_state.code_chunks)),
                               format_func=lambda i: f"{st.session_state.code_chunks[i]['file_path']} - {st.session_state.code_chunks[i]['entity_name']}")
    selected_chunk = st.session_state.code_chunks[chunk_index]
    st.write(f"File: {selected_chunk['file_path']}")
    st.write(f"Entity Type: {selected_chunk['entity_type']}")
    st.write(f"Entity Name: {selected_chunk['entity_name']}")
    st.write(f"Lines: {selected_chunk['start_line']} - {selected_chunk['end_line']}")
    st.code(selected_chunk['code'])
else:
    st.write("No code chunks available. Please preprocess the codebase first.")

# Add a section to display main files
st.header("- *Main Files*")
if st.session_state.main_files:
    selected_file = st.selectbox("Select a main file:", list(st.session_state.main_files.keys()))
    st.write(f"File: {selected_file}")
    st.code(st.session_state.main_files[selected_file])
else:
    st.write("No main files (main.py, script.py, app.py) found in the repository.")

# Add a section for additional information or statistics
st.header("- *Repository Statistics*")
if st.session_state.code_chunks:
    st.write(f"Total number of code chunks: {len(st.session_state.code_chunks)}")
    st.write(f"Total number of embeddings: {len(st.session_state.embeddings)}")
    file_count = len(set(chunk['file_path'] for chunk in st.session_state.code_chunks))
    st.write(f"Number of Python files processed: {file_count}")

    # Count of functions and classes
    function_count = sum(1 for chunk in st.session_state.code_chunks if chunk['entity_type'] == 'function')
    class_count = sum(1 for chunk in st.session_state.code_chunks if chunk['entity_type'] == 'class')
    st.write(f"Number of functions: {function_count}")
    st.write(f"Number of classes: {class_count}")

    # Display a pie chart of functions vs classes
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Pie(labels=['Functions', 'Classes'], values=[function_count, class_count])])
    fig.update_layout(title='Distribution of Functions and Classes')
    st.plotly_chart(fig)
else:
    st.write("No statistics available. Please preprocess the codebase first.")

# Add a footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and ChromaDB")