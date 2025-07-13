<<<<<<< HEAD
import warnings
from typing import Any

import lancedb
import numpy as np
import pandas as pd
from lancedb.db import DBConnection
from lancedb.rerankers import CrossEncoderReranker
from lancedb.rerankers.base import Reranker
from lancedb.table import Table

# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Configuration (Mirrors ingestion script and notebook) ---
LANCEDB_URI = "./vectordb/nutrition_facts"  # Local LanceDB path
TABLE_NAME = "nutrition_facts_notebook_style" # Table created by data_ingestion.py
EMBEDDING_DEVICE = "cpu"  # or "cuda" if available

# Retriever Configuration (Based on notebook/src defaults)
RETRIEVER_CONFIG = {
    "reranker": {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", # Common default cross-encoder
        "device": EMBEDDING_DEVICE,
    },
    "n_retrieve": 15, # Number of chunks to retrieve initially
    "n_titles": 7,    # Number of top titles to keep after grouping
    "enrich_first": True, # Whether to enrich the top result with surrounding chunks
    "window_size": 1 # Window size for enrichment if enrich_first is True
}

# --- Helper Functions (Adapted from src/retrieval.py) ---

def connect_to_lancedb_table(uri: str, table_name: str) -> Table:
    """Connects to a LanceDB database and opens the specified table."""
    print(f"Connecting to LanceDB at '{uri}' and opening table '{table_name}'...")
    try:
        db: DBConnection = lancedb.connect(uri=uri)
        table = db.open_table(table_name)
        print(f"Successfully opened table '{table_name}'. Schema: {table.schema}")
        print(f"Table contains {table.count_rows()} rows.")
        # Check for FTS index on 'text' which is needed for hybrid search
        # This is a basic check; LanceDB python API for checking indexes is limited
        # We assume the index exists if hybrid search doesn't fail later
        return table
    except Exception as e:
        print(f"Error connecting to or opening LanceDB table: {e}")
        raise # Re-raise the exception to halt execution if connection fails

def retrieve_context(
    k_base: Table,
    query_text: str,
    reranker_config: dict,
    n_retrieve: int = 10,
) -> list[dict]:
    """
    Retrieve relevant text chunks using hybrid search and rerank with a cross-encoder.
    """
    print(f"Retrieving top {n_retrieve} chunks for query: '{query_text}' using hybrid search...")

    # Configure reranker from the provided dictionary
    device: str = reranker_config.get("device", "cpu")
    rr_model_name: str = reranker_config.get("model_name")
    if not rr_model_name:
        raise ValueError("Reranker model name must be specified in reranker_config")

    print(f"Using reranker: {rr_model_name} on device: {device}")
    try:
        rr_cross_encoder: Reranker = CrossEncoderReranker(model_name=rr_model_name, device=device)
    except Exception as e:
        print(f"Error initializing CrossEncoderReranker: {e}")
        raise

    # Perform hybrid search and rerank
    try:
        search_result = k_base.search(query=query_text, query_type="hybrid")
        # Check if FTS index likely exists by checking if results are returned
        # A more robust check would involve trying a pure FTS search first if needed
        # peek_results = search_result.limit(1).to_list() # Temporarily removed peek
        # if not peek_results:
        #     print("Warning: Hybrid search returned no initial results. FTS index might be missing or query has no matches.")
        #     # Optionally, fall back to vector search or handle error
        #     # return [] # Example: return empty if no results

        reranked_results = search_result.rerank(reranker=rr_cross_encoder).limit(n_retrieve).to_list()
        print(f"Retrieved and reranked {len(reranked_results)} chunks.")
        return reranked_results
    except Exception as e:
        print(f"Error during LanceDB search or reranking: {e}")
        print("Ensure the table exists, has data, and necessary indexes (vector and FTS on 'text') are created.")
        # Example: Check if FTS index exists (requires table schema knowledge)
        if "text" not in k_base.schema.names:
             print("Error: 'text' field not found in table schema, required for FTS.")
        # Consider adding index creation here if it's missing, though it's better done in ingestion
        # try:
        #     print("Attempting to create FTS index on 'text'...")
        #     k_base.create_fts_index(["text"], replace=True) # Only index 'text'
        #     print("FTS index created. Please re-run the script.")
        # except Exception as index_e:
        #     print(f"Failed to create FTS index: {index_e}")
        raise # Re-raise the exception

def group_chunks_by_title(resp: list[dict], n_titles: int = 5) -> list[dict]:
    """
    Group retrieved text chunks by title, aggregate information, and return top titles.
    """
    if not resp:
        print("No chunks to group.")
        return []

    print(f"Grouping {len(resp)} chunks by title and selecting top {n_titles} titles...")
    # Required columns check
    required_cols = {"title", "url", "text", "_relevance_score"}
    if not all(col in resp[0] for col in required_cols):
        missing = required_cols - set(resp[0].keys())
        print(f"Warning: Missing required columns for grouping: {missing}. Using defaults.")
        # Add defaults if possible, or raise error
        for r in resp:
            r.setdefault("title", "Unknown Title")
            r.setdefault("url", "Unknown URL")
            r.setdefault("text", "")
            r.setdefault("_relevance_score", 0.0)
            # Add dummy values for optional cols if needed by agg
            r.setdefault("hash_doc", hash(r["text"])) # Example dummy hash
            r.setdefault("hash_title", hash(r["title"])) # Example dummy hash
            r.setdefault("n_docs", 1) # Example dummy
            r.setdefault("rank_abs", 0) # Example dummy
            r.setdefault("rank_rel", 0.0) # Example dummy


    # Pandas Chain
    grouped_data = (
        pd.DataFrame(resp)
        # .drop_duplicates(subset="hash_doc") # Requires hash_doc column
        .groupby("title")
        .agg(
            # First entry aggregation for unique fields
            # hash_title=("hash_title", "first"),
            url=("url", "first"),
            # n_docs=("n_docs", "first"),
            # Collect lists of text and ranks
            chunks=("text", list),
            # rank_abs=("rank_abs", list),
            # rank_rel=("rank_rel", list),
            # Compute
            score_sum=("_relevance_score", "sum"),
            n_chunks=("text", "count"),
        )
        .reset_index()
        .sort_values(by="score_sum", ascending=False)
        # .assign(cum_count=lambda x: x["n_chunks"].cumsum()) # Requires n_chunks
        .iloc[:n_titles]
        .to_dict("records")
    )
    print(f"Grouped into {len(grouped_data)} titles.")
    return grouped_data


def format_context(resp: list[dict]) -> str:
    """
    Format the grouped and potentially enriched context into a readable string.
    """
    if not resp:
        return "No context found."

    print("Formatting context...")
    output_lines: list[str] = []
    # Overlap logic removed as enrichment handles context better if enabled
    # overlap: int = get_rag_config()["embeddings"]["overlap"] # Needs local config

    for i, row in enumerate(resp):
        title: str = row.get("title", "Unknown Title")
        url: str = row.get("url", "Unknown URL")
        output_lines.append(f"{i + 1}. Title: '{title}' (URL: {url})")

        chunks: list[str] = row.get("chunks", [])
        # rank_abs: list[int] = row.get("rank_abs", []) # Needed for overlap logic

        if not chunks:
            output_lines.append("\t- No chunks available.")
        else:
            # Simple formatting: list each chunk
            for chunk in chunks:
                 output_lines.append(f"\t- {chunk}")
            # Overlap logic (requires rank_abs):
            # previous_rank: int = -1
            # for r_current, chunk in zip(rank_abs, chunks):
            #     if r_current - previous_rank > 1:
            #         output_lines.append(f"\t- {chunk}")
            #     elif len(chunk) > overlap:
            #         output_lines[-1] += chunk[overlap:]
            #     previous_rank = r_current

    formatted_string = "\n".join(output_lines)
    print("Context formatted.")
    return formatted_string


def enrich_text_chunks(k_base: Table, chunks_of_title: dict[str, Any], window_size: int = 1) -> dict[str, Any]:
    """
    Sentence-window retrieval: Fetch surrounding text chunks for a given title.
    NOTE: This function requires 'rank_abs', 'hash_title', and 'n_docs' columns
          in the LanceDB table, which might not exist in the simplified schema.
          This function might fail or return unchanged data if these columns are missing.
    """
    print(f"Attempting to enrich chunks for title: '{chunks_of_title.get('title', 'N/A')}' with window size {window_size}...")

    # Check for required keys from the grouped data
    required_keys = {"rank_abs", "hash_title", "chunks"}
    if not required_keys.issubset(chunks_of_title.keys()):
        print(f"Warning: Cannot enrich chunks. Missing required keys: {required_keys - set(chunks_of_title.keys())}")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    original_ranks: list[int] = chunks_of_title["rank_abs"]
    if not original_ranks:
        print("Warning: No original ranks found for enrichment.")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    # Generate ranks for chunks within the window
    new_ranks: set[int] = set()
    for r in original_ranks:
        for step in range(1, window_size + 1):
            if r - step >= 0: # Ensure rank is not negative
                 new_ranks.add(r - step)
            new_ranks.add(r + step) # Upper bound check happens during query
    new_ranks.difference_update(original_ranks)

    if not new_ranks:
        print("No new ranks to fetch for enrichment.")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    hash_title: str = chunks_of_title["hash_title"]
    new_ranks_str: str = ",".join(map(str, sorted(new_ranks)))
    query_text: str = f"hash_title = '{hash_title}' AND rank_abs IN ({new_ranks_str})"
    print(f"Enrichment query: {query_text}")

    # Check if required columns exist in the table schema for the query
    required_schema_fields = {"hash_title", "rank_abs", "text", "n_docs"}
    if not required_schema_fields.issubset(k_base.schema.names):
         print(f"Warning: Cannot execute enrichment query. Table schema missing required fields: {required_schema_fields - set(k_base.schema.names)}")
         chunks_of_title["enriched"] = False
         return chunks_of_title

    try:
        fields: list[str] = ["rank_abs", "text", "n_docs"]
        new_chunks_df: pd.DataFrame = k_base.search().where(query_text).limit(len(new_ranks)*2).to_pandas()[fields] # Limit added for safety
    except Exception as e:
        print(f"Error during enrichment query: {e}")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    if new_chunks_df.empty:
        print("No additional chunks found during enrichment.")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    print(f"Found {len(new_chunks_df)} additional chunks for enrichment.")

    # Combine original and new chunks
    text_dict: dict[int, str] = dict(zip(original_ranks, chunks_of_title["chunks"]))
    text_dict.update({c["rank_abs"]: c["text"] for c in new_chunks_df.to_dict("records")})

    # Update the dictionary with enriched data
    enriched_chunk_info: dict[str, Any] = chunks_of_title.copy()
    updated_ranks: list[int] = sorted(text_dict)
    enriched_chunk_info["rank_abs"] = updated_ranks
    enriched_chunk_info["chunks"] = [text_dict[r] for r in updated_ranks]
    enriched_chunk_info["enriched"] = True

    # Update other fields if they exist
    if "n_chunks" in enriched_chunk_info:
        enriched_chunk_info["n_chunks"] = len(updated_ranks)
    # n_docs should be consistent for the title
    n_docs_val = new_chunks_df["n_docs"].iloc[0] if not new_chunks_df.empty else chunks_of_title.get("n_docs", 1)
    if "rank_rel" in enriched_chunk_info:
         enriched_chunk_info["rank_rel"] = (np.array(updated_ranks) / n_docs_val).tolist() if n_docs_val else [0.0] * len(updated_ranks)
    if "cum_count" in enriched_chunk_info:
        enriched_chunk_info["cum_count"] += len(new_chunks_df)

    print("Enrichment successful.")
    return enriched_chunk_info


def get_context(
    k_base: Table,
    query_text: str,
    reranker_config: dict,
    n_retrieve: int = 10,
    n_titles: int = 5,
    enrich_first: bool = False,
    window_size: int = 1,
) -> str:
    """
    Retrieve, group, optionally enrich, and format context for a given query.
    """
    print("-" * 20)
    print("Starting context retrieval process...")
    # 1. Retrieve raw context with reranking
    cxt_raw: list[dict] = retrieve_context(
        k_base=k_base,
        query_text=query_text,
        reranker_config=reranker_config,
        n_retrieve=n_retrieve
    )

    if not cxt_raw:
        print("No raw context retrieved.")
        return "No context found for the query."

    # 2. Group retrieved chunks by title
    cxt_grouped: list[dict] = group_chunks_by_title(cxt_raw, n_titles=n_titles)

    if not cxt_grouped:
        print("Context grouping resulted in empty list.")
        return "No context found after grouping."

    # 3. Optionally enrich the first title
    if enrich_first:
        print("Enrichment requested for the top title.")
        # Note: Enrichment requires specific columns (hash_title, rank_abs, n_docs)
        # which might not be present in the simplified schema.
        cxt_grouped[0] = enrich_text_chunks(
            k_base=k_base,
            chunks_of_title=cxt_grouped[0],
            window_size=window_size
        )
    else:
        print("Enrichment not requested.")

    # 4. Format the final context
    cxt_string: str = format_context(cxt_grouped)
    print("Context retrieval process complete.")
    print("-" * 20)
    return cxt_string

# --- Main Execution ---

if __name__ == "__main__":
    # Connect to the knowledge base
    try:
        knowledge_base = connect_to_lancedb_table(LANCEDB_URI, TABLE_NAME)
    except Exception as e:
        print(f"Failed to initialize knowledge base. Exiting. Error: {e}")
        exit()

    # Example Query
    # query = "How to reduce heart Disease Risk"
    query = "Benefits of turmeric"
    # query = "Are energy drinks risky?"

    print(f"\nExecuting retrieval for query: '{query}'")

    # Get context using the defined configuration
    final_context = get_context(
        k_base=knowledge_base,
        query_text=query,
        reranker_config=RETRIEVER_CONFIG["reranker"],
        n_retrieve=RETRIEVER_CONFIG["n_retrieve"],
        n_titles=RETRIEVER_CONFIG["n_titles"],
        enrich_first=RETRIEVER_CONFIG["enrich_first"], # Set to True to test enrichment
        window_size=RETRIEVER_CONFIG["window_size"]
    )

    print("\n--- Final Context ---")
    print(final_context)
    print("--- End of Context ---")
=======
import warnings
from typing import Any

import lancedb
import numpy as np
import pandas as pd
from lancedb.db import DBConnection
from lancedb.rerankers import CrossEncoderReranker
from lancedb.rerankers.base import Reranker
from lancedb.table import Table

# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Configuration (Mirrors ingestion script and notebook) ---
LANCEDB_URI = "./vectordb/nutrition_facts"  # Local LanceDB path
TABLE_NAME = "nutrition_facts_notebook_style" # Table created by data_ingestion.py
EMBEDDING_DEVICE = "cpu"  # or "cuda" if available

# Retriever Configuration (Based on notebook/src defaults)
RETRIEVER_CONFIG = {
    "reranker": {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", # Common default cross-encoder
        "device": EMBEDDING_DEVICE,
    },
    "n_retrieve": 15, # Number of chunks to retrieve initially
    "n_titles": 7,    # Number of top titles to keep after grouping
    "enrich_first": True, # Whether to enrich the top result with surrounding chunks
    "window_size": 1 # Window size for enrichment if enrich_first is True
}

# --- Helper Functions (Adapted from src/retrieval.py) ---

def connect_to_lancedb_table(uri: str, table_name: str) -> Table:
    """Connects to a LanceDB database and opens the specified table."""
    print(f"Connecting to LanceDB at '{uri}' and opening table '{table_name}'...")
    try:
        db: DBConnection = lancedb.connect(uri=uri)
        table = db.open_table(table_name)
        print(f"Successfully opened table '{table_name}'. Schema: {table.schema}")
        print(f"Table contains {table.count_rows()} rows.")
        # Check for FTS index on 'text' which is needed for hybrid search
        # This is a basic check; LanceDB python API for checking indexes is limited
        # We assume the index exists if hybrid search doesn't fail later
        return table
    except Exception as e:
        print(f"Error connecting to or opening LanceDB table: {e}")
        raise # Re-raise the exception to halt execution if connection fails

def retrieve_context(
    k_base: Table,
    query_text: str,
    reranker_config: dict,
    n_retrieve: int = 10,
) -> list[dict]:
    """
    Retrieve relevant text chunks using hybrid search and rerank with a cross-encoder.
    """
    print(f"Retrieving top {n_retrieve} chunks for query: '{query_text}' using hybrid search...")

    # Configure reranker from the provided dictionary
    device: str = reranker_config.get("device", "cpu")
    rr_model_name: str = reranker_config.get("model_name")
    if not rr_model_name:
        raise ValueError("Reranker model name must be specified in reranker_config")

    print(f"Using reranker: {rr_model_name} on device: {device}")
    try:
        rr_cross_encoder: Reranker = CrossEncoderReranker(model_name=rr_model_name, device=device)
    except Exception as e:
        print(f"Error initializing CrossEncoderReranker: {e}")
        raise

    # Perform hybrid search and rerank
    try:
        search_result = k_base.search(query=query_text, query_type="hybrid")
        # Check if FTS index likely exists by checking if results are returned
        # A more robust check would involve trying a pure FTS search first if needed
        # peek_results = search_result.limit(1).to_list() # Temporarily removed peek
        # if not peek_results:
        #     print("Warning: Hybrid search returned no initial results. FTS index might be missing or query has no matches.")
        #     # Optionally, fall back to vector search or handle error
        #     # return [] # Example: return empty if no results

        reranked_results = search_result.rerank(reranker=rr_cross_encoder).limit(n_retrieve).to_list()
        print(f"Retrieved and reranked {len(reranked_results)} chunks.")
        return reranked_results
    except Exception as e:
        print(f"Error during LanceDB search or reranking: {e}")
        print("Ensure the table exists, has data, and necessary indexes (vector and FTS on 'text') are created.")
        # Example: Check if FTS index exists (requires table schema knowledge)
        if "text" not in k_base.schema.names:
             print("Error: 'text' field not found in table schema, required for FTS.")
        # Consider adding index creation here if it's missing, though it's better done in ingestion
        # try:
        #     print("Attempting to create FTS index on 'text'...")
        #     k_base.create_fts_index(["text"], replace=True) # Only index 'text'
        #     print("FTS index created. Please re-run the script.")
        # except Exception as index_e:
        #     print(f"Failed to create FTS index: {index_e}")
        raise # Re-raise the exception

def group_chunks_by_title(resp: list[dict], n_titles: int = 5) -> list[dict]:
    """
    Group retrieved text chunks by title, aggregate information, and return top titles.
    """
    if not resp:
        print("No chunks to group.")
        return []

    print(f"Grouping {len(resp)} chunks by title and selecting top {n_titles} titles...")
    # Required columns check
    required_cols = {"title", "url", "text", "_relevance_score"}
    if not all(col in resp[0] for col in required_cols):
        missing = required_cols - set(resp[0].keys())
        print(f"Warning: Missing required columns for grouping: {missing}. Using defaults.")
        # Add defaults if possible, or raise error
        for r in resp:
            r.setdefault("title", "Unknown Title")
            r.setdefault("url", "Unknown URL")
            r.setdefault("text", "")
            r.setdefault("_relevance_score", 0.0)
            # Add dummy values for optional cols if needed by agg
            r.setdefault("hash_doc", hash(r["text"])) # Example dummy hash
            r.setdefault("hash_title", hash(r["title"])) # Example dummy hash
            r.setdefault("n_docs", 1) # Example dummy
            r.setdefault("rank_abs", 0) # Example dummy
            r.setdefault("rank_rel", 0.0) # Example dummy


    # Pandas Chain
    grouped_data = (
        pd.DataFrame(resp)
        # .drop_duplicates(subset="hash_doc") # Requires hash_doc column
        .groupby("title")
        .agg(
            # First entry aggregation for unique fields
            # hash_title=("hash_title", "first"),
            url=("url", "first"),
            # n_docs=("n_docs", "first"),
            # Collect lists of text and ranks
            chunks=("text", list),
            # rank_abs=("rank_abs", list),
            # rank_rel=("rank_rel", list),
            # Compute
            score_sum=("_relevance_score", "sum"),
            n_chunks=("text", "count"),
        )
        .reset_index()
        .sort_values(by="score_sum", ascending=False)
        # .assign(cum_count=lambda x: x["n_chunks"].cumsum()) # Requires n_chunks
        .iloc[:n_titles]
        .to_dict("records")
    )
    print(f"Grouped into {len(grouped_data)} titles.")
    return grouped_data


def format_context(resp: list[dict]) -> str:
    """
    Format the grouped and potentially enriched context into a readable string.
    """
    if not resp:
        return "No context found."

    print("Formatting context...")
    output_lines: list[str] = []
    # Overlap logic removed as enrichment handles context better if enabled
    # overlap: int = get_rag_config()["embeddings"]["overlap"] # Needs local config

    for i, row in enumerate(resp):
        title: str = row.get("title", "Unknown Title")
        url: str = row.get("url", "Unknown URL")
        output_lines.append(f"{i + 1}. Title: '{title}' (URL: {url})")

        chunks: list[str] = row.get("chunks", [])
        # rank_abs: list[int] = row.get("rank_abs", []) # Needed for overlap logic

        if not chunks:
            output_lines.append("\t- No chunks available.")
        else:
            # Simple formatting: list each chunk
            for chunk in chunks:
                 output_lines.append(f"\t- {chunk}")
            # Overlap logic (requires rank_abs):
            # previous_rank: int = -1
            # for r_current, chunk in zip(rank_abs, chunks):
            #     if r_current - previous_rank > 1:
            #         output_lines.append(f"\t- {chunk}")
            #     elif len(chunk) > overlap:
            #         output_lines[-1] += chunk[overlap:]
            #     previous_rank = r_current

    formatted_string = "\n".join(output_lines)
    print("Context formatted.")
    return formatted_string


def enrich_text_chunks(k_base: Table, chunks_of_title: dict[str, Any], window_size: int = 1) -> dict[str, Any]:
    """
    Sentence-window retrieval: Fetch surrounding text chunks for a given title.
    NOTE: This function requires 'rank_abs', 'hash_title', and 'n_docs' columns
          in the LanceDB table, which might not exist in the simplified schema.
          This function might fail or return unchanged data if these columns are missing.
    """
    print(f"Attempting to enrich chunks for title: '{chunks_of_title.get('title', 'N/A')}' with window size {window_size}...")

    # Check for required keys from the grouped data
    required_keys = {"rank_abs", "hash_title", "chunks"}
    if not required_keys.issubset(chunks_of_title.keys()):
        print(f"Warning: Cannot enrich chunks. Missing required keys: {required_keys - set(chunks_of_title.keys())}")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    original_ranks: list[int] = chunks_of_title["rank_abs"]
    if not original_ranks:
        print("Warning: No original ranks found for enrichment.")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    # Generate ranks for chunks within the window
    new_ranks: set[int] = set()
    for r in original_ranks:
        for step in range(1, window_size + 1):
            if r - step >= 0: # Ensure rank is not negative
                 new_ranks.add(r - step)
            new_ranks.add(r + step) # Upper bound check happens during query
    new_ranks.difference_update(original_ranks)

    if not new_ranks:
        print("No new ranks to fetch for enrichment.")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    hash_title: str = chunks_of_title["hash_title"]
    new_ranks_str: str = ",".join(map(str, sorted(new_ranks)))
    query_text: str = f"hash_title = '{hash_title}' AND rank_abs IN ({new_ranks_str})"
    print(f"Enrichment query: {query_text}")

    # Check if required columns exist in the table schema for the query
    required_schema_fields = {"hash_title", "rank_abs", "text", "n_docs"}
    if not required_schema_fields.issubset(k_base.schema.names):
         print(f"Warning: Cannot execute enrichment query. Table schema missing required fields: {required_schema_fields - set(k_base.schema.names)}")
         chunks_of_title["enriched"] = False
         return chunks_of_title

    try:
        fields: list[str] = ["rank_abs", "text", "n_docs"]
        new_chunks_df: pd.DataFrame = k_base.search().where(query_text).limit(len(new_ranks)*2).to_pandas()[fields] # Limit added for safety
    except Exception as e:
        print(f"Error during enrichment query: {e}")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    if new_chunks_df.empty:
        print("No additional chunks found during enrichment.")
        chunks_of_title["enriched"] = False
        return chunks_of_title

    print(f"Found {len(new_chunks_df)} additional chunks for enrichment.")

    # Combine original and new chunks
    text_dict: dict[int, str] = dict(zip(original_ranks, chunks_of_title["chunks"]))
    text_dict.update({c["rank_abs"]: c["text"] for c in new_chunks_df.to_dict("records")})

    # Update the dictionary with enriched data
    enriched_chunk_info: dict[str, Any] = chunks_of_title.copy()
    updated_ranks: list[int] = sorted(text_dict)
    enriched_chunk_info["rank_abs"] = updated_ranks
    enriched_chunk_info["chunks"] = [text_dict[r] for r in updated_ranks]
    enriched_chunk_info["enriched"] = True

    # Update other fields if they exist
    if "n_chunks" in enriched_chunk_info:
        enriched_chunk_info["n_chunks"] = len(updated_ranks)
    # n_docs should be consistent for the title
    n_docs_val = new_chunks_df["n_docs"].iloc[0] if not new_chunks_df.empty else chunks_of_title.get("n_docs", 1)
    if "rank_rel" in enriched_chunk_info:
         enriched_chunk_info["rank_rel"] = (np.array(updated_ranks) / n_docs_val).tolist() if n_docs_val else [0.0] * len(updated_ranks)
    if "cum_count" in enriched_chunk_info:
        enriched_chunk_info["cum_count"] += len(new_chunks_df)

    print("Enrichment successful.")
    return enriched_chunk_info


def get_context(
    k_base: Table,
    query_text: str,
    reranker_config: dict,
    n_retrieve: int = 10,
    n_titles: int = 5,
    enrich_first: bool = False,
    window_size: int = 1,
) -> str:
    """
    Retrieve, group, optionally enrich, and format context for a given query.
    """
    print("-" * 20)
    print("Starting context retrieval process...")
    # 1. Retrieve raw context with reranking
    cxt_raw: list[dict] = retrieve_context(
        k_base=k_base,
        query_text=query_text,
        reranker_config=reranker_config,
        n_retrieve=n_retrieve
    )

    if not cxt_raw:
        print("No raw context retrieved.")
        return "No context found for the query."

    # 2. Group retrieved chunks by title
    cxt_grouped: list[dict] = group_chunks_by_title(cxt_raw, n_titles=n_titles)

    if not cxt_grouped:
        print("Context grouping resulted in empty list.")
        return "No context found after grouping."

    # 3. Optionally enrich the first title
    if enrich_first:
        print("Enrichment requested for the top title.")
        # Note: Enrichment requires specific columns (hash_title, rank_abs, n_docs)
        # which might not be present in the simplified schema.
        cxt_grouped[0] = enrich_text_chunks(
            k_base=k_base,
            chunks_of_title=cxt_grouped[0],
            window_size=window_size
        )
    else:
        print("Enrichment not requested.")

    # 4. Format the final context
    cxt_string: str = format_context(cxt_grouped)
    print("Context retrieval process complete.")
    print("-" * 20)
    return cxt_string

# --- Main Execution ---

if __name__ == "__main__":
    # Connect to the knowledge base
    try:
        knowledge_base = connect_to_lancedb_table(LANCEDB_URI, TABLE_NAME)
    except Exception as e:
        print(f"Failed to initialize knowledge base. Exiting. Error: {e}")
        exit()

    # Example Query
    # query = "How to reduce heart Disease Risk"
    query = "Benefits of turmeric"
    # query = "Are energy drinks risky?"

    print(f"\nExecuting retrieval for query: '{query}'")

    # Get context using the defined configuration
    final_context = get_context(
        k_base=knowledge_base,
        query_text=query,
        reranker_config=RETRIEVER_CONFIG["reranker"],
        n_retrieve=RETRIEVER_CONFIG["n_retrieve"],
        n_titles=RETRIEVER_CONFIG["n_titles"],
        enrich_first=RETRIEVER_CONFIG["enrich_first"], # Set to True to test enrichment
        window_size=RETRIEVER_CONFIG["window_size"]
    )

    print("\n--- Final Context ---")
    print(final_context)
    print("--- End of Context ---")
>>>>>>> d4f67a4f568fcb30cdec9b3d88002221f673e55e
