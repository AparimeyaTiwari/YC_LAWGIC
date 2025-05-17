# LAWGIC ⚖️ – AI-Powered Indian Legal Law Agent

LAWGIC is a modular, multilingual AI-powered legal assistant tailored for the Indian legal ecosystem. Designed with accessibility, clarity, and ethical AI principles in mind, LAWGIC helps users understand, query, and navigate legal documents and laws with ease.

Whether you're reading a lease agreement, seeking clarity on IPC sections, or looking for a lawyer nearby, LAWGIC simplifies the experience using natural language, multimodal input, and smart semantic retrieval.

### ✅ Real-World Relevance
LAWGIC addresses critical gaps in the Indian legal system—especially the inaccessibility of legal knowledge to the common citizen, language barriers, and lack of structured legal search. It's built to support real use cases like contract review, case law navigation, and lawyer discovery.

### 🧠 Human-in-the-Loop Support
LAWGIC encourages responsible use by integrating optional *Human-in-the-Loop* workflows, enabling users to consult verified lawyers and validate AI-generated legal insights for sensitive or high-impact scenarios.

### 🛡️ Responsible AI by Design
LAWGIC follows Responsible AI principles including:
- **Transparency**: Retrieval grounding from legal documents.
- **Multilingual Fairness**: Support for all major Indian languages.
- **Traceability**: Explainable outputs with references to legal sources.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [Modular Plugins with Semantic Kernel](#-modular-plugins-with-semantic-kernel)
3. [Architecture Overview](#architecture-overview)
4. [Software Stack](#software-stack)
5. [Environment Variables](#-environment-variables)
6. [Setup & Development](#-setup--development)
7. [Usage Guide](#-usage-guide)
8. [What's Next?](#-whats-next)
9. [Demonstration](#-Demonstration)
11. [Contact](#-contact)

---

## 🧩 Key Features

### 📁 Multi-Modal Input
- Accepts **text**, **PDF**, **images**, and soon **speech input**.
- Uses **Azure Computer Vision OCR** to extract text from uploaded files.
- Supports input in **any Indian or global language**.

### 🌐 Multilingual Translation
- Automatically detects the input language.
- Translates to **English** using **Azure Translator** for internal processing.
- Final output is translated **back to the original language** for user-friendly delivery.

### 🧠 AI-Powered Legal Insights
- Translated input is vectorized and compared with **Azure AI Search indexes**.
- Legal documents are stored in **Azure Blob Storage**, indexed semantically.
- Google Drive Folder – Legal Documents to Upload in Blob Storage - https://drive.google.com/drive/folders/1Nh4a6g5XYf_LyhLgbZQ-bLTeaDpCyB-m
- Retrieved content is passed to **Azure OpenAI LLM** for legal reasoning and answer generation.
- Ensures accuracy and context relevance by grounding answers in legal documents.

### 📍 Nearby Legal Help
- Provides a list of **nearby lawyers** using **Google Maps API**.
- Supports three location modes: `IP Address`, `Geocode`, and `Manual Input`.

---

## 🧠 Modular Plugins with Semantic Kernel
LAWGIC leverages **Semantic Kernel** to orchestrate specialized plugins that work together to fulfill complex legal tasks.

### 🔁 Translator Plugin
- Handles **language detection** and **bi-directional translation** (Input to English, Output to Original).
- Uses **Azure Translator** under the hood.

### ⚖️ Lawyer AI Plugin
- Converts translated query to vector using **Azure OpenAI Embeddings**.
- Searches semantic index from **Azure AI Search** linked to PDFs in **Azure Blob**.
- Sends retrieved context to **Azure LLM** (ChatGPT) to generate an accurate, grounded legal response.

### 📍 Nearby Lawyers Plugin
- Processes user’s IP, geocode, or manual input to determine location.
- Fetches relevant legal professionals nearby via **Google Maps API**.
- Presents result directly in the Chainlit UI.

---

## 🏗️ Architecture Overview

![image](https://github.com/user-attachments/assets/a459b063-d9a5-40db-a526-cb99c89a5545)

*Figure: End-to-end LAWGIC Architecture*

Text-to-Speech (TTS) and Speech-to-Text (STT) functionalities are planned as upcoming features.


### Input Handling:
- `Text`, `Image`, and `PDF` inputs → processed using **Azure Computer Vision**.
- Optional: Future integration with **Azure Speech-to-Text**.

### Processing Pipeline:
1. Extracted/typed input is translated to English.
2. Query is converted to vector using Azure Embeddings.
3. Azure AI Search compares it to document indexes.
4. Top-matching content passed to Azure LLM.
5. Answer generated and translated back to original language.

### Output:
- Delivered in native language.
- Cleanly presented via **Chainlit UI**.

---

## 🖥️ Software Stack

### Frontend:
- **Chainlit** (Python-based UI)

### Backend & AI Orchestration:
- **Python**
- **Semantic Kernel (Plugins + Planner)**
- **Azure OpenAI Service** (Embeddings + LLM)
- **Azure Computer Vision** (OCR)
- **Azure Translator** (Language Translation)
- **Google Maps API** (Nearby Lawyer Search)

### Storage:
- **Azure Blob Storage** (Legal Document PDFs)
- **Azure AI Search** (Indexing & Semantic Search)

---

## 🔧 Environment Variables
Before running the system, configure the following:

```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT="https://<your-search-resource>.search.windows.net"
AZURE_SEARCH_API_KEY="<your-azure-search-api-key>"
AZURE_SEARCH_INDEX_NAME_1="azureblob-index"
AZURE_SEARCH_INDEX_NAME_2="azureblob-index-2"
AZURE_SEARCH_INDEX_NAME_3="azureblob-index-3"

# Azure OpenAI (LLM) -> for LLM 1
ENDPOINT_URL="https://<your-openai-resource>.openai.azure.com/"
AZURE_OPENAI_API_KEY_1="<your-azure-openai-key>"
DEPLOYMENT_NAME="<your model>"


# Azure OpenAI (Embeddings)
EMBEDDING_EP="https://<your-openai-resource>.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
KEY="<your-embedding-api-key>"
EMB_DEPLoy="text-embedding-ada-002"

# Azure OpenAI (LLM) -> for LLM 2
AZURE_OPENAI_ENDPOINT="https://<your-openai-resource>.openai.azure.com/"
AZURE_OPENAI_API_KEY="<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT_NAME="<your deployment name>"
AZURE_OPENAI_API_VERSION="<your api version"

# Azure Translator
TRANSLATOR_KEY="<your-translator-key>"
TRANSLATOR_REGION="<your region"
TRANSLATOR_ENDPOINT="https://api.cognitive.microsofttranslator.com/"

# Google Maps API
GOOGLE_MAPS_API_KEY="<your-google-maps-api-key>"

# Chainlit Authentication
CHAINLIT_AUTH_SECRET="<your-chainlit-auth-secret>"
OAUTH_GOOGLE_CLIENT_ID="<your-google-client-id>"
OAUTH_GOOGLE_CLIENT_SECRET="<your-google-client-secret>"
CHAINLIT_URL="http://localhost:8000"

#OCR keys
VISION_KEY = "<your_vision_key">
VISION_ENDPOINT = "<your_vision_endpoint">
# Optional: Plugin Info Token
PINFO_TOKEN="<your-plugin-info-token>"

```

---

## 🛠️ Setup & Development

### Prerequisites:
- Python >= 3.10
- Azure resources (OpenAI, Computer Vision, Translator, AI Search, Blob Storage)

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/lawgic.git
cd lawgic
```

### 2. Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
- Create a `.env` file using the template and insert your Azure & Google credentials.

### 5. Run the App
```bash
chainlit run main.py --port 8000
```

---

## 🚀 Usage Guide

### Step 1: Input Your Query
- Upload a document OR type your legal question.
- Supports Indian languages and English.

### Step 2: Let LAWGIC Analyze
- Text is extracted (if needed) and translated.
- Semantic search locates relevant laws.
- Azure LLM generates an appropriate legal response.

### Step 3: Get Results
- The answer is translated back to your preferred language.
- Displayed with legal references (if available).
- Use the **Nearby Lawyer** button to find help locally.

---

## 📌 What's Next?
- User profiles for history and saved queries.
- Role-based filters for specific legal domains.
- Public-facing legal FAQ database with AI curation.

---

## 📽️ Demonstration
- Youtube Link: 


---

## 📬 Contact
Have questions or want to collaborate?
- Aparimeya Tiwari: https://www.linkedin.com/in/aparimeya-tiwari-76a252252/
- Vinayak Khavare: https://www.linkedin.com/in/vinayak-khavare-542821257/
- Richa Rathi: https://www.linkedin.com/in/richa-rathi-775871257/
- Sahil Adit: https://www.linkedin.com/in/sahiladit/


# YC_LAWGIC
