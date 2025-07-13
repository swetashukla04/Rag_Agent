<<<<<<< HEAD
import json
import os
import re
import warnings
from pathlib import Path

import lancedb
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer

# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Configuration (Hardcoded for Simplicity)
LANCEDB_URI = "./vectordb/nutrition_facts"  # Local LanceDB path
POST_JSON_PATH = "./data/blog_posts/json"  # Path to JSON files
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_DEVICE = "cpu"  # or "cuda" if available
TABLE_NAME = "nutrition_facts_notebook_style" # Use a distinct table name

# Chunking Configuration (based on notebook defaults/derivations)
# Assuming n_token_max = 256 from notebook context if get_rag_config was used
N_TOKEN_MAX = 256
N_CHAR_MAX = N_TOKEN_MAX * 4  # Approximate character count based on tokens
OVERLAP = int(N_CHAR_MAX * 0.1) # 10% overlap

# Load environment variables (e.g., for potential future API use)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # Not strictly needed for local SentenceTransformer via registry

# --- Helper Functions (Integrated from notebook/src) ---

def recursive_text_splitter(text: str, n_char_max: int = 1000, overlap: int = 100) -> list[str]:
    """
    Helper function for chunking text recursively.
    Splits a long text into smaller chunks based on max characters and overlap.
    Aims to split preferably at separators like newlines and spaces.
    """
    if len(text) <= n_char_max:
        return [text]

    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + n_char_max, len(text))
        chunk = text[start_idx:end_idx]

        # If not the last chunk, try to find a good split point before the end
        if end_idx < len(text):
            # Prefer splitting by double newline, then single newline, then space
            split_pos = -1
            for sep in ["\n\n", "\n", " ", ""]:
                # Search backwards from the end index
                temp_pos = chunk.rfind(sep, max(0, start_idx), end_idx)
                if temp_pos != -1 and temp_pos > start_idx: # Ensure split is not at the very beginning
                    split_pos = temp_pos + len(sep) # Include separator in the split point logic
                    break
            if split_pos != -1:
                end_idx = split_pos

        # Adjust chunk based on potentially found split point
        chunk = text[start_idx:end_idx]
        chunks.append(chunk.strip()) # Add the chunk

        # Determine the start of the next chunk with overlap
        next_start_idx = end_idx - overlap
        # Ensure next start index is not before the current start index if overlap is large
        if next_start_idx <= start_idx:
             next_start_idx = start_idx + 1 # Move forward by at least one char

        # If overlap pushes start beyond text length, break
        if next_start_idx >= len(text):
            break

        start_idx = next_start_idx
        # Safety break to prevent infinite loops on edge cases
        if start_idx >= end_idx:
             break

    # Clean up empty strings that might result from splitting
    return [c for c in chunks if c]


def text_has_only_questions(text: str) -> bool:
    """
    Returns True if the input string contains question marks but not periods or exclamation marks.
    (aka text consists of only questions but no information)
    """
    return "?" in text and "." not in text and "!" not in text

# --- LanceDB Setup ---

# Get embedding function definition from LanceDB registry
embedding_func = get_registry().get("sentence-transformers").create(name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)

# Define the data model using LanceModel and the embedding function
class DataModel(LanceModel):
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
    text: str = embedding_func.SourceField() # Field to embed
    title: str
    url: str
    blog_tags: str

# --- Main Ingestion Logic ---

if __name__ == "__main__":
    print("Starting data ingestion (notebook style)...")

    # 1. Load Tokenizer (needed for chunking logic based on tokens)
    print(f"Loading tokenizer for: {EMBEDDING_MODEL_NAME}")
    try:
        tokenizer = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE).tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Proceeding without token-based length check.")
        tokenizer = None # Fallback if tokenizer loading fails

    # 2. Data Processing and Chunking
    print(f"Processing JSON files from: {POST_JSON_PATH}")
    files: list[Path] = list(Path(POST_JSON_PATH).glob("*.json"))
    data_to_ingest = []

    for json_file in files:
        print(f"Processing: {json_file.name}")
        try:
            with open(json_file, encoding='utf-8') as f: # Specify encoding
                doc: dict = json.load(f)

            paragraphs: list[str] = doc.get("paragraphs", [])
            title: str = doc.get("title", "N/A")
            url: str = doc.get("url", "N/A")
            # Ensure blog_tags is a list of strings before joining
            raw_tags = doc.get("blog_tags", [])
            blog_tags: str = " ".join(set(str(tag) for tag in raw_tags if isinstance(tag, str)))

            processed_paragraphs: list[str] = []
            for i, para in enumerate(paragraphs):
                if not isinstance(para, str) or not para.strip(): # Skip non-strings or empty paragraphs
                    continue

                if text_has_only_questions(para):
                    # print(f"  Skipping paragraph {i} (question only).")
                    continue

                # Check length - prefer character length check now
                if len(para) > N_CHAR_MAX:
                    # print(f"  Paragraph {i}: {len(para)} chars > {N_CHAR_MAX}. Splitting needed.")
                    para_chunks: list[str] = recursive_text_splitter(para, N_CHAR_MAX, OVERLAP)
                    # print(f"    Split into {len(para_chunks)} chunks.")
                    processed_paragraphs.extend(para_chunks)
                else:
                    processed_paragraphs.append(para)

            # Create records for LanceDB
            for chunk_text in processed_paragraphs:
                 data_to_ingest.append({
                     "text": chunk_text,
                     "title": title,
                     "url": url,
                     "blog_tags": blog_tags
                 })

        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON in {json_file.name}: {e}")
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")

    if not data_to_ingest:
        print("No data processed. Exiting.")
        exit()

    print(f"\nProcessed {len(data_to_ingest)} text chunks from {len(files)} files.")

    # 3. LanceDB Ingestion
    print(f"Connecting/Creating LanceDB database at: {LANCEDB_URI}")
    db = lancedb.connect(LANCEDB_URI)

    print(f"Creating table '{TABLE_NAME}' (mode: overwrite)...")
    try:
        # Create table using the schema which defines embedding
        table = db.create_table(TABLE_NAME, schema=DataModel, mode="overwrite")
    except Exception as e:
        print(f"Error creating table: {e}")
        exit()

    print(f"Adding {len(data_to_ingest)} records to table (embedding handled by LanceDB)...")
    try:
        # Add data directly; LanceDB uses the schema to embed the 'text' field into 'vector'
        table.add(data_to_ingest)
        print(f"Successfully ingested {table.count_rows()} entries into table '{TABLE_NAME}'.")
    except Exception as e:
        print(f"Error adding data to table: {e}")
        exit()

    # 4. Create Indexes (Vector and FTS)
    print("\nCreating vector index (metric: cosine)...")
    try:
        # Create a vector index (e.g., IVF_PQ) for faster vector search
        # LanceDB might create a default index, but explicit creation allows configuration
        table.create_index(metric="cosine", replace=True) # Using cosine as in notebook example
        print("Vector index created successfully.")
    except Exception as e:
        print(f"Error creating vector index: {e}")
        # Decide if this is critical; maybe vector search still works without explicit index

    print("\nCreating FTS index on 'text' column...")
    try:
        # Create FTS index (inverted index) required for hybrid search
        table.create_fts_index(["text"], replace=True) # Indexing only 'text' column for FTS
        print("FTS index created successfully.")
    except Exception as e:
        print(f"Error creating FTS index: {e}")
        # FTS index is crucial for the current retrieval script's hybrid search
        print("FTS index creation failed. Hybrid search in retrieval.py will likely fail.")

    print("\nData ingestion and indexing complete!")
=======
import json
import os
import re
import warnings
from pathlib import Path

import lancedb
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer

# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Configuration (Hardcoded for Simplicity)
LANCEDB_URI = "./vectordb/nutrition_facts"  # Local LanceDB path
POST_JSON_PATH = "./data/blog_posts/json"  # Path to JSON files
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_DEVICE = "cpu"  # or "cuda" if available
TABLE_NAME = "nutrition_facts_notebook_style" # Use a distinct table name

# Chunking Configuration (based on notebook defaults/derivations)
# Assuming n_token_max = 256 from notebook context if get_rag_config was used
N_TOKEN_MAX = 256
N_CHAR_MAX = N_TOKEN_MAX * 4  # Approximate character count based on tokens
OVERLAP = int(N_CHAR_MAX * 0.1) # 10% overlap

# Load environment variables (e.g., for potential future API use)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # Not strictly needed for local SentenceTransformer via registry

# --- Helper Functions (Integrated from notebook/src) ---

def recursive_text_splitter(text: str, n_char_max: int = 1000, overlap: int = 100) -> list[str]:
    """
    Helper function for chunking text recursively.
    Splits a long text into smaller chunks based on max characters and overlap.
    Aims to split preferably at separators like newlines and spaces.
    """
    if len(text) <= n_char_max:
        return [text]

    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + n_char_max, len(text))
        chunk = text[start_idx:end_idx]

        # If not the last chunk, try to find a good split point before the end
        if end_idx < len(text):
            # Prefer splitting by double newline, then single newline, then space
            split_pos = -1
            for sep in ["\n\n", "\n", " ", ""]:
                # Search backwards from the end index
                temp_pos = chunk.rfind(sep, max(0, start_idx), end_idx)
                if temp_pos != -1 and temp_pos > start_idx: # Ensure split is not at the very beginning
                    split_pos = temp_pos + len(sep) # Include separator in the split point logic
                    break
            if split_pos != -1:
                end_idx = split_pos

        # Adjust chunk based on potentially found split point
        chunk = text[start_idx:end_idx]
        chunks.append(chunk.strip()) # Add the chunk

        # Determine the start of the next chunk with overlap
        next_start_idx = end_idx - overlap
        # Ensure next start index is not before the current start index if overlap is large
        if next_start_idx <= start_idx:
             next_start_idx = start_idx + 1 # Move forward by at least one char

        # If overlap pushes start beyond text length, break
        if next_start_idx >= len(text):
            break

        start_idx = next_start_idx
        # Safety break to prevent infinite loops on edge cases
        if start_idx >= end_idx:
             break

    # Clean up empty strings that might result from splitting
    return [c for c in chunks if c]


def text_has_only_questions(text: str) -> bool:
    """
    Returns True if the input string contains question marks but not periods or exclamation marks.
    (aka text consists of only questions but no information)
    """
    return "?" in text and "." not in text and "!" not in text

# --- LanceDB Setup ---

# Get embedding function definition from LanceDB registry
embedding_func = get_registry().get("sentence-transformers").create(name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)

# Define the data model using LanceModel and the embedding function
class DataModel(LanceModel):
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
    text: str = embedding_func.SourceField() # Field to embed
    title: str
    url: str
    blog_tags: str

# --- Main Ingestion Logic ---

if __name__ == "__main__":
    print("Starting data ingestion (notebook style)...")

    # 1. Load Tokenizer (needed for chunking logic based on tokens)
    print(f"Loading tokenizer for: {EMBEDDING_MODEL_NAME}")
    try:
        tokenizer = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE).tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Proceeding without token-based length check.")
        tokenizer = None # Fallback if tokenizer loading fails

    # 2. Data Processing and Chunking
    print(f"Processing JSON files from: {POST_JSON_PATH}")
    files: list[Path] = list(Path(POST_JSON_PATH).glob("*.json"))
    data_to_ingest = []

    for json_file in files:
        print(f"Processing: {json_file.name}")
        try:
            with open(json_file, encoding='utf-8') as f: # Specify encoding
                doc: dict = json.load(f)

            paragraphs: list[str] = doc.get("paragraphs", [])
            title: str = doc.get("title", "N/A")
            url: str = doc.get("url", "N/A")
            # Ensure blog_tags is a list of strings before joining
            raw_tags = doc.get("blog_tags", [])
            blog_tags: str = " ".join(set(str(tag) for tag in raw_tags if isinstance(tag, str)))

            processed_paragraphs: list[str] = []
            for i, para in enumerate(paragraphs):
                if not isinstance(para, str) or not para.strip(): # Skip non-strings or empty paragraphs
                    continue

                if text_has_only_questions(para):
                    # print(f"  Skipping paragraph {i} (question only).")
                    continue

                # Check length - prefer character length check now
                if len(para) > N_CHAR_MAX:
                    # print(f"  Paragraph {i}: {len(para)} chars > {N_CHAR_MAX}. Splitting needed.")
                    para_chunks: list[str] = recursive_text_splitter(para, N_CHAR_MAX, OVERLAP)
                    # print(f"    Split into {len(para_chunks)} chunks.")
                    processed_paragraphs.extend(para_chunks)
                else:
                    processed_paragraphs.append(para)

            # Create records for LanceDB
            for chunk_text in processed_paragraphs:
                 data_to_ingest.append({
                     "text": chunk_text,
                     "title": title,
                     "url": url,
                     "blog_tags": blog_tags
                 })

        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON in {json_file.name}: {e}")
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")

    if not data_to_ingest:
        print("No data processed. Exiting.")
        exit()

    print(f"\nProcessed {len(data_to_ingest)} text chunks from {len(files)} files.")

    # 3. LanceDB Ingestion
    print(f"Connecting/Creating LanceDB database at: {LANCEDB_URI}")
    db = lancedb.connect(LANCEDB_URI)

    print(f"Creating table '{TABLE_NAME}' (mode: overwrite)...")
    try:
        # Create table using the schema which defines embedding
        table = db.create_table(TABLE_NAME, schema=DataModel, mode="overwrite")
    except Exception as e:
        print(f"Error creating table: {e}")
        exit()

    print(f"Adding {len(data_to_ingest)} records to table (embedding handled by LanceDB)...")
    try:
        # Add data directly; LanceDB uses the schema to embed the 'text' field into 'vector'
        table.add(data_to_ingest)
        print(f"Successfully ingested {table.count_rows()} entries into table '{TABLE_NAME}'.")
    except Exception as e:
        print(f"Error adding data to table: {e}")
        exit()

    # 4. Create Indexes (Vector and FTS)
    print("\nCreating vector index (metric: cosine)...")
    try:
        # Create a vector index (e.g., IVF_PQ) for faster vector search
        # LanceDB might create a default index, but explicit creation allows configuration
        table.create_index(metric="cosine", replace=True) # Using cosine as in notebook example
        print("Vector index created successfully.")
    except Exception as e:
        print(f"Error creating vector index: {e}")
        # Decide if this is critical; maybe vector search still works without explicit index

    print("\nCreating FTS index on 'text' column...")
    try:
        # Create FTS index (inverted index) required for hybrid search
        table.create_fts_index(["text"], replace=True) # Indexing only 'text' column for FTS
        print("FTS index created successfully.")
    except Exception as e:
        print(f"Error creating FTS index: {e}")
        # FTS index is crucial for the current retrieval script's hybrid search
        print("FTS index creation failed. Hybrid search in retrieval.py will likely fail.")

    print("\nData ingestion and indexing complete!")
>>>>>>> d4f67a4f568fcb30cdec9b3d88002221f673e55e
