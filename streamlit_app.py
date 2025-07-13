<<<<<<< HEAD
import streamlit as st
import warnings
import google.generativeai as genai
import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
st.set_page_config(page_title="Nutrition Facts RAG", layout="wide")


# Import functions from our retrieval and rag_flow scripts using relative import
# Assuming streamlit_app.py is in the same directory as retrieval.py and rag_flow.py
try:
    from simple_rag_app.retrieval import connect_to_lancedb_table, get_context, RETRIEVER_CONFIG, LANCEDB_URI, TABLE_NAME
    # We need the prompt template and LLM query function from rag_flow
    from simple_rag_app.rag_flow import SYSTEM_PROMPT_TEMPLATE, query_llm, WELCOME_MESSAGE
except ImportError:
    st.error("Could not import necessary functions from simple_rag_app.retrieval.py or simple_rag_app.rag_flow.py. Make sure they are in the simple_rag_app directory.")
    st.stop()


# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Configuration ---
# Hardcode your Gemini API Key here (Consider using Streamlit secrets for production)
# GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Example using secrets
GOOGLE_API_KEY = "AIzaSyCqtWK2u5IXQqisW7GRIuK2bLlYOLTQRxQ" # Replace with your actual key

if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY:
    st.warning("Warning: GOOGLE_API_KEY is not set. Please configure it (e.g., using Streamlit secrets or replacing the placeholder). LLM query will fail.", icon="⚠️")
    # Don't exit in Streamlit, just show warning

# LLM Configuration
LLM_MODEL = "gemini-1.5-flash" # Use a model compatible with the API key
LLM_TEMPERATURE = 0.5 # Temperature is often set during model generation call in genai

# --- Initialization --- (Cache expensive operations)

@st.cache_resource # Cache the DB connection
def initialize_knowledge_base():
    """Connects to the LanceDB table."""
    try:
        knowledge_base = connect_to_lancedb_table(LANCEDB_URI, TABLE_NAME)
        return knowledge_base
    except Exception as e:
        st.error(f"Failed to initialize knowledge base. Error: {e}")
        return None

@st.cache_resource # Cache the LLM client
def initialize_llm_client(api_key, model_name):
    """Initializes the Gemini GenerativeModel."""
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        st.error("Cannot initialize LLM client without a valid GOOGLE_API_KEY.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini model. Error: {e}")
        return None

knowledge_base = initialize_knowledge_base()
gemini_model = initialize_llm_client(GOOGLE_API_KEY, LLM_MODEL)

# --- Streamlit UI ---

# st.set_page_config(page_title="Nutrition Facts RAG", layout="wide")
st.title("🍎 Nutrition Facts RAG Assistant")
st.markdown(f"> {WELCOME_MESSAGE}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
             with st.expander("View Retrieved Context"):
                 st.caption(message["context"])


# Accept user input
if prompt := st.chat_input("Ask your nutrition question..."):
    if not knowledge_base:
        st.error("Knowledge base is not available. Cannot process query.")
    elif not gemini_model:
        st.error("LLM client is not available. Cannot process query.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display thinking indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            # 1. Retrieve Context
            context_str = get_context(
                k_base=knowledge_base,
                query_text=prompt,
                reranker_config=RETRIEVER_CONFIG["reranker"],
                n_retrieve=RETRIEVER_CONFIG["n_retrieve"],
                n_titles=RETRIEVER_CONFIG["n_titles"],
                enrich_first=RETRIEVER_CONFIG["enrich_first"],
                window_size=RETRIEVER_CONFIG["window_size"]
            )

            if not context_str or context_str == "No context found for the query.":
                response = "I could not find relevant information in the knowledge base to answer your question."
                context_str = None # Ensure context expander doesn't show
            else:
                # 2. Build Prompt
                system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
                # Construct messages for the LLM (simplified for Gemini's text-only input)
                llm_prompt_text = f"{system_content}\n\nUser Query: {prompt}"

                # 3. Query LLM
                try:
                    # Note: query_llm from rag_flow expects a list of dicts,
                    # but gemini's generate_content takes a string directly.
                    # We'll call generate_content directly here for simplicity.
                    llm_response_obj = gemini_model.generate_content(
                        llm_prompt_text,
                        generation_config=genai.types.GenerationConfig(
                            temperature=LLM_TEMPERATURE
                        )
                    )
                    response = llm_response_obj.text
                except Exception as e:
                    st.error(f"Error querying LLM: {e}")
                    response = "Sorry, I encountered an error trying to generate a response."
                    context_str = None # Hide context if LLM fails

            # Display assistant response
            message_placeholder.markdown(response)
            if context_str:
                with st.expander("View Retrieved Context"):
                    st.caption(context_str)

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "context": context_str # Store context with the message
        })

else:
    # Display initial instructions or clear history button if needed
    if not st.session_state.messages:
        st.info("Enter your query in the box below to get started.")
=======
import streamlit as st
import warnings
import google.generativeai as genai
import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
st.set_page_config(page_title="Nutrition Facts RAG", layout="wide")


# Import functions from our retrieval and rag_flow scripts using relative import
# Assuming streamlit_app.py is in the same directory as retrieval.py and rag_flow.py
try:
    from simple_rag_app.retrieval import connect_to_lancedb_table, get_context, RETRIEVER_CONFIG, LANCEDB_URI, TABLE_NAME
    # We need the prompt template and LLM query function from rag_flow
    from simple_rag_app.rag_flow import SYSTEM_PROMPT_TEMPLATE, query_llm, WELCOME_MESSAGE
except ImportError:
    st.error("Could not import necessary functions from simple_rag_app.retrieval.py or simple_rag_app.rag_flow.py. Make sure they are in the simple_rag_app directory.")
    st.stop()


# Ignore specific warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Configuration ---
# Hardcode your Gemini API Key here (Consider using Streamlit secrets for production)
# GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Example using secrets
GOOGLE_API_KEY = "AIzaSyCqtWK2u5IXQqisW7GRIuK2bLlYOLTQRxQ" # Replace with your actual key

if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GOOGLE_API_KEY:
    st.warning("Warning: GOOGLE_API_KEY is not set. Please configure it (e.g., using Streamlit secrets or replacing the placeholder). LLM query will fail.", icon="⚠️")
    # Don't exit in Streamlit, just show warning

# LLM Configuration
LLM_MODEL = "gemini-1.5-flash" # Use a model compatible with the API key
LLM_TEMPERATURE = 0.5 # Temperature is often set during model generation call in genai

# --- Initialization --- (Cache expensive operations)

@st.cache_resource # Cache the DB connection
def initialize_knowledge_base():
    """Connects to the LanceDB table."""
    try:
        knowledge_base = connect_to_lancedb_table(LANCEDB_URI, TABLE_NAME)
        return knowledge_base
    except Exception as e:
        st.error(f"Failed to initialize knowledge base. Error: {e}")
        return None

@st.cache_resource # Cache the LLM client
def initialize_llm_client(api_key, model_name):
    """Initializes the Gemini GenerativeModel."""
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        st.error("Cannot initialize LLM client without a valid GOOGLE_API_KEY.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini model. Error: {e}")
        return None

knowledge_base = initialize_knowledge_base()
gemini_model = initialize_llm_client(GOOGLE_API_KEY, LLM_MODEL)

# --- Streamlit UI ---

# st.set_page_config(page_title="Nutrition Facts RAG", layout="wide")
st.title("🍎 Nutrition Facts RAG Assistant")
st.markdown(f"> {WELCOME_MESSAGE}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
             with st.expander("View Retrieved Context"):
                 st.caption(message["context"])


# Accept user input
if prompt := st.chat_input("Ask your nutrition question..."):
    if not knowledge_base:
        st.error("Knowledge base is not available. Cannot process query.")
    elif not gemini_model:
        st.error("LLM client is not available. Cannot process query.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display thinking indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            # 1. Retrieve Context
            context_str = get_context(
                k_base=knowledge_base,
                query_text=prompt,
                reranker_config=RETRIEVER_CONFIG["reranker"],
                n_retrieve=RETRIEVER_CONFIG["n_retrieve"],
                n_titles=RETRIEVER_CONFIG["n_titles"],
                enrich_first=RETRIEVER_CONFIG["enrich_first"],
                window_size=RETRIEVER_CONFIG["window_size"]
            )

            if not context_str or context_str == "No context found for the query.":
                response = "I could not find relevant information in the knowledge base to answer your question."
                context_str = None # Ensure context expander doesn't show
            else:
                # 2. Build Prompt
                system_content = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
                # Construct messages for the LLM (simplified for Gemini's text-only input)
                llm_prompt_text = f"{system_content}\n\nUser Query: {prompt}"

                # 3. Query LLM
                try:
                    # Note: query_llm from rag_flow expects a list of dicts,
                    # but gemini's generate_content takes a string directly.
                    # We'll call generate_content directly here for simplicity.
                    llm_response_obj = gemini_model.generate_content(
                        llm_prompt_text,
                        generation_config=genai.types.GenerationConfig(
                            temperature=LLM_TEMPERATURE
                        )
                    )
                    response = llm_response_obj.text
                except Exception as e:
                    st.error(f"Error querying LLM: {e}")
                    response = "Sorry, I encountered an error trying to generate a response."
                    context_str = None # Hide context if LLM fails

            # Display assistant response
            message_placeholder.markdown(response)
            if context_str:
                with st.expander("View Retrieved Context"):
                    st.caption(context_str)

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "context": context_str # Store context with the message
        })

else:
    # Display initial instructions or clear history button if needed
    if not st.session_state.messages:
        st.info("Enter your query in the box below to get started.")
>>>>>>> d4f67a4f568fcb30cdec9b3d88002221f673e55e
