# config/settings.py
import os

# Azure OpenAI settings – update these with your actual values or set as environment variables.
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://akuai.openai.azure.com/")
OPENAI_KEY = os.getenv("OPENAI_KEY", "ec1d7395a75c480d9bc0d1dec7e9d331")
OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT", "aku-text-embedding-ada-002")
OPENAI_COMPLETIONS_DEPLOYMENT = os.getenv("OPENAI_COMPLETIONS_DEPLOYMENT", "gpt-4o-mini-emploeeassistant")

# Cosmos DB settings – update with your account info.
COSMOS_URI = os.getenv("COSMOS_URI", "https://akurag1.documents.azure.com:443/")
COSMOS_KEY = os.getenv("COSMOS_KEY", "rsICHsUdvbEuN0g4NoiIuJMTmzDFKYlpbtUAMe3ohXIFqwXMqb9eWCPHhR4UBD2GEGHbNe6VX6Z4ACDbh2PY7w==")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "policy_db")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "policy_vector_2")
# COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "policy_kb_v2")

COSMOS_FEEDBACK_URI = os.getenv("COSMOS_FEEDBACK_URI", "https://aku-chatbot-feedback.documents.azure.com:443/")
COSMOS_FEEDBACK_KEY= os.getenv("COSMOS_FEEDBACK_KEY", "ul7T17HrjHgxtdPFUZuUZ8TESyOnRmZUL7RMydbVUGNyRGnWnaHTMC2uafiNnVmK5psG9R1JI3pzACDbAKuAPw==")

# Optionally, PDF_DIR if you plan to process PDFs.
# Get the absolute path to the "data" folder inside the "config" folder.


# PDF_DIR = os.getenv("PDF_DIR", "pdfs")

PDF_DIR = r"C:\Users\hussain.baig\OneDrive - Aga Khan University\Desktop\Clinic Assistant\Clinic Assistant - Backend\data"

FAQ_DIR = "data/faqs"


FORM_RECOGNIZER_ENDPOINT = "https://docu-int-2025.cognitiveservices.azure.com/"
FORM_RECOGNIZER_KEY      = "A6e1R9sb0495XfE9qHQc1gLFKRhHYEQkLortiU3ieyQCn0vwMox7JQQJ99BGACYeBjFXJ3w3AAALACOGCvLb"


CLINICAL_INSTRUCTIONS_DIR = r"C:\Users\hussain.baig\OneDrive - Aga Khan University\Desktop\Clinic Assistant\Clinic Assistant - Backend\data\Clinical_Instructions"

