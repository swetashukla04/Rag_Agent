<<<<<<< HEAD
import warnings
# import os # No longer needed for getenv
# from dotenv import load_dotenv # No longer using .env
# from groq import Groq # No longer using Groq
import google.generativeai as genai
import requests # For checking models, optional

# Import functions from our retrieval script using relative import
from .retrieval import connect_to_lancedb_table, get_context, RETRIEVER_CONFIG, LANCEDB_URI, TABLE_NAME

# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Configuration ---
# Hardcode your Gemini API Key here
GOOGLE_API_KEY = "AIzaSyCqtWK2u5IXQqisW7GRIuK2bLlYOLTQRxQ" # Replace with your actual key

if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set in the script. Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual key. LLM query will fail.")
    # exit() # Or handle gracefully

# LLM Configuration
LLM_MODEL = "gemini-2.0-flash" # Or other available Gemini model
LLM_TEMPERATURE = 0.5

# --- Prompt Templates ---
# Simplified from the notebook's prompt building
# SYSTEM_PROMPT_TEMPLATE = """
# You are a knowledgeable and helpful assistant trained to answer questions about nutrition, health, and diet based on evidence from NutritionFacts.org.

# Use *only* the information provided in the context below to answer the user's question. Be clear, concise, and specific in your response.

# - If the context provides a complete answer, respond directly and helpfully.
# - If the context gives partial information, summarize what is available and suggest relevant topics the user might explore further.
# - If the context does not contain enough information to answer the question, say so clearly without making anything up.

# Do not fabricate information. Always ground your response in the given context.

# Context:
# ---
# {context}
# ---
# """


SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable and helpful assistant trained to answer questions about nutrition, health, and diet based on evidence from NutritionFacts.org.

Context:
---
{context}
---
"""

WELCOME_MESSAGE = "Hello! How can I help you with your nutrition questions today?"

# --- LLM Interaction Functions ---

# def get_groq_models(api_key: str) -> list[str]: # No longer needed
#     """Optional: Fetches available Groq models."""
#     models_url = "https://api.groq.com/openai/v1/models"
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     try:
#         response = requests.get(url=models_url, headers=headers, timeout=10)
#         response.raise_for_status()
#         model_data = response.json().get("data", [])
#         active_models = sorted([md["id"] for md in model_data if md.get("active")])
#         return active_models
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching Groq models: {e}")
#         return []

def query_llm(model: genai.GenerativeModel, messages: list[dict]) -> str:
    """Sends the prompt to the Gemini API and returns the response content."""
    print("\nQuerying LLM (Gemini)...")
    try:
        prompt_text = ""
        for message in messages:
            if message["role"] == "user":
                prompt_text += message["content"]
            elif message["role"] == "system":
                prompt_text = message["content"] + "\n" + prompt_text # System prompt first

        response = model.generate_content(prompt_text)
        content = response.text
        # usage = dict(response.usage) # Gemini doesn't directly expose usage like Groq
        # print(f"LLM Usage: {usage}")
        return content
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return "Error: Could not get response from LLM."

# --- Main RAG Flow ---

if __name__ == "__main__":
    print("Starting RAG Flow...")

    # 1. Setup: Connect to DB and LLM Client
    try:
        knowledge_base = connect_to_lancedb_table(LANCEDB_URI, TABLE_NAME)
    except Exception as e:
        print(f"Failed to initialize knowledge base. Exiting. Error: {e}")
        exit()

    if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set. Cannot proceed with LLM query. Exiting.")
        exit()

    # Initialize Gemini model
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(LLM_MODEL)

    # Optional: Check available models (Not directly supported in this example)
    # available_models = get_groq_models(GROQ_API_KEY)
    # print("\nAvailable Groq Models:")
    # for m in available_models: print(f"- {m}")
    # if LLM_MODEL not in available_models:
    #     print(f"Warning: Configured model '{LLM_MODEL}' might not be available/active.")

    # 2. Get User Query
    # query = "How can I reduce my heart Disease Risk?"
    query = "What are the benefits of turmeric for arsenic exposure?"
    # query = "Tell me about vitamin B12 dosage"
    print(f"\nUser Query: {query}")

    # 3. Retrieve Context
    print("\nRetrieving context...")
    context_str = get_context(
        k_base=knowledge_base,
        query_text=query,
        reranker_config=RETRIEVER_CONFIG["reranker"],
        n_retrieve=RETRIEVER_CONFIG["n_retrieve"],
        n_titles=RETRIEVER_CONFIG["n_titles"],
        enrich_first=RETRIEVER_CONFIG["enrich_first"],
        window_size=RETRIEVER_CONFIG["window_size"]
    )

    if not context_str or context_str == "No context found for the query.":
        print("Could not retrieve relevant context. Cannot proceed with LLM query.")
        # Optionally, you could still query the LLM without context,
        # but that defeats the purpose of RAG.
        exit()

    # print("\n--- Retrieved Context ---")
    # print(context_str)
    # print("--- End of Context ---")

    # 4. Build Prompt
    print("\nBuilding prompt...")
    system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
    messages = [
        {"role": "system", "content": system_content},
        # {"role": "assistant", "content": WELCOME_MESSAGE}, # Optional welcome
        {"role": "user", "content": query},
    ]
    # print("\n--- Prompt Messages ---")
    # for msg in messages: print(f"{msg['role'].upper()}: {msg['content'][:200]}...") # Print truncated prompt
    # print("--- End of Prompt ---")


    # 5. Query LLM
    llm_response = query_llm(
        model=gemini_model,
        messages=messages,
    )

    # 6. Display Result
    print("\n--- LLM Response ---")
    print(llm_response)
    print("--- End of Response ---")

    print("\nRAG Flow Complete.")
=======
import warnings
# import os # No longer needed for getenv
# from dotenv import load_dotenv # No longer using .env
# from groq import Groq # No longer using Groq
import google.generativeai as genai
import requests # For checking models, optional

# Import functions from our retrieval script using relative import
from .retrieval import connect_to_lancedb_table, get_context, RETRIEVER_CONFIG, LANCEDB_URI, TABLE_NAME

# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Configuration ---
# Hardcode your Gemini API Key here
GOOGLE_API_KEY = "AIzaSyCqtWK2u5IXQqisW7GRIuK2bLlYOLTQRxQ" # Replace with your actual key

if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set in the script. Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual key. LLM query will fail.")
    # exit() # Or handle gracefully

# LLM Configuration
LLM_MODEL = "gemini-2.0-flash" # Or other available Gemini model
LLM_TEMPERATURE = 0.5

# --- Prompt Templates ---
# Simplified from the notebook's prompt building
# SYSTEM_PROMPT_TEMPLATE = """
# You are a knowledgeable and helpful assistant trained to answer questions about nutrition, health, and diet based on evidence from NutritionFacts.org.

# Use *only* the information provided in the context below to answer the user's question. Be clear, concise, and specific in your response.

# - If the context provides a complete answer, respond directly and helpfully.
# - If the context gives partial information, summarize what is available and suggest relevant topics the user might explore further.
# - If the context does not contain enough information to answer the question, say so clearly without making anything up.

# Do not fabricate information. Always ground your response in the given context.

# Context:
# ---
# {context}
# ---
# """


SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable and helpful assistant trained to answer questions about nutrition, health, and diet based on evidence from NutritionFacts.org.

Context:
---
{context}
---
"""

WELCOME_MESSAGE = "Hello! How can I help you with your nutrition questions today?"

# --- LLM Interaction Functions ---

# def get_groq_models(api_key: str) -> list[str]: # No longer needed
#     """Optional: Fetches available Groq models."""
#     models_url = "https://api.groq.com/openai/v1/models"
#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     try:
#         response = requests.get(url=models_url, headers=headers, timeout=10)
#         response.raise_for_status()
#         model_data = response.json().get("data", [])
#         active_models = sorted([md["id"] for md in model_data if md.get("active")])
#         return active_models
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching Groq models: {e}")
#         return []

def query_llm(model: genai.GenerativeModel, messages: list[dict]) -> str:
    """Sends the prompt to the Gemini API and returns the response content."""
    print("\nQuerying LLM (Gemini)...")
    try:
        prompt_text = ""
        for message in messages:
            if message["role"] == "user":
                prompt_text += message["content"]
            elif message["role"] == "system":
                prompt_text = message["content"] + "\n" + prompt_text # System prompt first

        response = model.generate_content(prompt_text)
        content = response.text
        # usage = dict(response.usage) # Gemini doesn't directly expose usage like Groq
        # print(f"LLM Usage: {usage}")
        return content
    except Exception as e:
        print(f"Error querying Gemini API: {e}")
        return "Error: Could not get response from LLM."

# --- Main RAG Flow ---

if __name__ == "__main__":
    print("Starting RAG Flow...")

    # 1. Setup: Connect to DB and LLM Client
    try:
        knowledge_base = connect_to_lancedb_table(LANCEDB_URI, TABLE_NAME)
    except Exception as e:
        print(f"Failed to initialize knowledge base. Exiting. Error: {e}")
        exit()

    if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set. Cannot proceed with LLM query. Exiting.")
        exit()

    # Initialize Gemini model
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(LLM_MODEL)

    # Optional: Check available models (Not directly supported in this example)
    # available_models = get_groq_models(GROQ_API_KEY)
    # print("\nAvailable Groq Models:")
    # for m in available_models: print(f"- {m}")
    # if LLM_MODEL not in available_models:
    #     print(f"Warning: Configured model '{LLM_MODEL}' might not be available/active.")

    # 2. Get User Query
    # query = "How can I reduce my heart Disease Risk?"
    query = "What are the benefits of turmeric for arsenic exposure?"
    # query = "Tell me about vitamin B12 dosage"
    print(f"\nUser Query: {query}")

    # 3. Retrieve Context
    print("\nRetrieving context...")
    context_str = get_context(
        k_base=knowledge_base,
        query_text=query,
        reranker_config=RETRIEVER_CONFIG["reranker"],
        n_retrieve=RETRIEVER_CONFIG["n_retrieve"],
        n_titles=RETRIEVER_CONFIG["n_titles"],
        enrich_first=RETRIEVER_CONFIG["enrich_first"],
        window_size=RETRIEVER_CONFIG["window_size"]
    )

    if not context_str or context_str == "No context found for the query.":
        print("Could not retrieve relevant context. Cannot proceed with LLM query.")
        # Optionally, you could still query the LLM without context,
        # but that defeats the purpose of RAG.
        exit()

    # print("\n--- Retrieved Context ---")
    # print(context_str)
    # print("--- End of Context ---")

    # 4. Build Prompt
    print("\nBuilding prompt...")
    system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
    messages = [
        {"role": "system", "content": system_content},
        # {"role": "assistant", "content": WELCOME_MESSAGE}, # Optional welcome
        {"role": "user", "content": query},
    ]
    # print("\n--- Prompt Messages ---")
    # for msg in messages: print(f"{msg['role'].upper()}: {msg['content'][:200]}...") # Print truncated prompt
    # print("--- End of Prompt ---")


    # 5. Query LLM
    llm_response = query_llm(
        model=gemini_model,
        messages=messages,
    )

    # 6. Display Result
    print("\n--- LLM Response ---")
    print(llm_response)
    print("--- End of Response ---")

    print("\nRAG Flow Complete.")
>>>>>>> d4f67a4f568fcb30cdec9b3d88002221f673e55e
