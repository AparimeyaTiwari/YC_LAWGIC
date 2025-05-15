import chainlit as cl
from chainlit.types import AskSpec
import os
import time
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import googlemaps
import requests
from typing import Optional
import docx
from io import BytesIO
import PyPDF2
import requests
import uuid
from os import environ
from typing import List, Dict, Any
from datetime import datetime
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import asyncio
import re
import os
import chainlit as cl
from dotenv import load_dotenv
import requests
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from langdetect import detect, LangDetectException

# Load prompts
with open("prompt.txt", "r") as f:
    system_prompt = f.read()

# Load environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY_1")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
index_name_1 = os.getenv("AZURE_SEARCH_INDEX_NAME_1")
index_name_2 = os.getenv("AZURE_SEARCH_INDEX_NAME_2")
index_name_3 = os.getenv("AZURE_SEARCH_INDEX_NAME_3")

EMBEDDINGEP = os.getenv("EMBEDDING_EP")
empkey = os.getenv("KEY")
empdep = os.getenv("EMB_DEPLoy")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {empkey}"
}

# Check for missing keys
if not endpoint or not deployment or not api_key or not EMBEDDINGEP or not empkey:
    print("Missing keys!!! Check your .env file")
    exit()

# Azure OpenAI Chat Client
chat_client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
)

# Simplified language options - only English and Hindi
language_options = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "marathi": "mr-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "urdu": "ur-IN"
}

# Mapping from language code to Azure voice names
language_to_voice = {
    'en': 'en-IN-NeerjaNeural',
    'hi': 'hi-IN-SwaraNeural',
    'bn': 'bn-IN-TanishaaNeural',
    'gu': 'gu-IN-DhwaniNeural',
    'kn': 'kn-IN-SapnaNeural',
    'ml': 'ml-IN-SobhanaNeural',
    'mr': 'mr-IN-AarohiNeural',
    'pa': 'pa-IN-GaganNeural',
    'ta': 'ta-IN-PallaviNeural',
    'te': 'te-IN-MohanNeural',
    'ur': 'ur-IN-SalmanNeural'
}

def get_speech_config(language_code="en-IN"):
    """Get Azure Speech SDK configuration"""
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    if not speech_key or not speech_region:
        raise RuntimeError("Azure credentials missing")
    config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    config.speech_recognition_language = language_code
    return config

async def recognize_speech(language_locale="en-IN"):
    """Recognize speech from microphone with specified language"""
    speech_config = get_speech_config(language_locale)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    try:
        loop = asyncio.get_running_loop()
        future = recognizer.recognize_once_async()
        result = await loop.run_in_executor(None, future.get)

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip(), language_locale[:2]  # Return text and language code
        elif result.reason == speechsdk.ResultReason.NoMatch:
            raise ValueError("No speech detected")
        else:
            raise ValueError(f"Recognition error: {result.cancellation_details.reason}")
    except Exception as e:
        raise ValueError(f"Speech recognition failed: {str(e)}")

async def synthesize_speech(text: str, language_code: str = None):
    """Convert text to speech with automatic language detection"""
    try:
        if not language_code:
            try:
                language_code = detect(text)
            except LangDetectException:
                language_code = 'en'  # Fallback to English
        
        voice_name = language_to_voice.get(language_code, 'en-IN-NeerjaNeural')
        
        speech_config = get_speech_config()
        speech_config.speech_synthesis_voice_name = voice_name
        # Use default speaker with async
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # Use async synthesis with proper await
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: synthesizer.speak_text(text))
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return True
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(f"TTS canceled: {details.reason} - {details.error_details}")
        return False
    except Exception as e:
        raise RuntimeError(f"TTS error: {str(e)}")

async def embedder(text: str):
    data = {
        "input": [text]
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": empkey
    }

    response = requests.post(EMBEDDINGEP, headers=headers, json=data)
    
    try:
        response.raise_for_status()
        resp_json = response.json()
        
        if "data" not in resp_json:
            raise ValueError(f"Invalid response format: {resp_json}")
        return resp_json['data'][0]['embedding']
    except:
        print(f"Error in embedding request: {response.status_code} - {response.text}")
        return None
    
def chunker(ip, chunk_size=10000):
    return [ip[i:i + chunk_size] for i in range(0, len(ip), chunk_size)]

def vector_search(query_embedding, embeddings):
    similarities = []
    for embed, chunk, act_name in embeddings:
        similarity_score = cosine_similarity([query_embedding], [embed])[0][0]
        similarities.append((similarity_score, chunk, act_name))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[0]

index_keywords = {
    "index1": ["finance", "security", "taxation", "corporate", "company", "environment", "land", "insurance", "civil", "personal law", "succession"],
    "index2": ["family", "governance", "criminal", "succession law", "personal law"],
    "index3": ["labour", "employment", "intellectual property"]
}

def detect_relevant_index(query: str):
    query_lower = query.lower()
    for index, keywords in index_keywords.items():
        for word in keywords:
            if word in query_lower:
                return index
    return None

# Initialize global variables
kernel = None
gmaps = None

# initialize translator key and endpoint
language_map = {
    'hi': 'Hindi',
    'en': 'English',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'pa': 'Punjabi',
    'ml': 'Malayalam',
    'or': 'Odia',
    'as': 'Assamese',
    'ur': 'Urdu',
    'kok': 'Konkani',
    'mai': 'Maithili',
    'ks': 'Kashmiri',
    'ne': 'Nepali',
    'sd': 'Sindhi',
    'sa': 'Sanskrit',
    'bho': 'Bhojpuri',
    'dog': 'Dogri',
    'mni': 'Manipuri',
    'sat': 'Santali',
}

key = os.getenv("TRANSLATOR_KEY")
tr_endpoint = os.getenv("TRANSLATOR_ENDPOINT")
region = os.getenv("TRANSLATOR_REGION")

headers = {
    'Ocp-Apim-Subscription-Key': key,
    'Ocp-Apim-Subscription-Region': region,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

subscription_key = os.getenv("VISION_KEY")
ocr_endpoint = os.getenv("VISION_ENDPOINT")

# Authenticate
computervision_client = ComputerVisionClient(ocr_endpoint, CognitiveServicesCredentials(subscription_key))

async def ocr(file_path: str) -> str:
    """
    Performs OCR on the given file using Azure Computer Vision and returns extracted text.
    
    Args:
        file_path: Path to the file to process (supports PDF, JPEG, PNG, TIFF, BMP)
        
    Returns:
        Extracted text concatenated from all pages
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If OCR processing fails
    """
    # Validate input file
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"OCR input file not found: {file_path}")
    
    if os.path.getsize(file_path) == 0:
        raise ValueError("OCR input file is empty")
    
    # Check supported file types
    valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    if not file_path.lower().endswith(tuple(valid_extensions)):
        raise ValueError(f"Unsupported file type. Supported formats: {', '.join(valid_extensions)}")

    try:
        # Send file to Azure Computer Vision
        with open(file_path, "rb") as local_file:
            read_response = computervision_client.read_in_stream(
                local_file, 
                raw=True,
                reading_order="natural"  # Preserve logical reading order
            )
        
        # Extract operation ID from response headers
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]
        
        # Poll for results with progress feedback
        start_time = time.time()
        max_wait_time = 120  # 2 minute timeout
        last_status = None
        
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            current_status = read_result.status.lower()
            
            # Send status updates if changed
            if current_status != last_status:
                print(f"OCR Status: {current_status}")
                last_status = current_status
            
            if current_status not in {'notstarted', 'running'}:
                break
                
            if time.time() - start_time > max_wait_time:
                raise TimeoutError("OCR processing timed out")
                
            await asyncio.sleep(1)  # Non-blocking sleep

        # Process results
        if read_result.status.lower() != 'succeeded':
            raise RuntimeError(f"OCR processing failed with status: {read_result.status}")
        
        # Extract and concatenate text with page separation
        extracted_text = []
        for page in read_result.analyze_result.read_results:
            page_text = " ".join(line.text for line in page.lines)
            extracted_text.append(page_text)
            
            # Add page break if multi-page document
            if len(read_result.analyze_result.read_results) > 1:
                extracted_text.append(f"\n[PAGE BREAK: Page {page.page}]\n")
        
        full_text = " ".join(extracted_text)
        
        # Basic post-processing
        full_text = re.sub(r'\s+', ' ', full_text).strip()  # Normalize whitespace
        
        print(f"Successfully extracted {len(full_text)} characters from {file_path}")
        return full_text
        
    except Exception as e:
        error_msg = f"OCR processing error: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: dict[str, str], default_user: cl.User) -> Optional[cl.User]:
    return default_user

async def detect_and_translate_to_english (text):
    # Step 1: Detect Language
    detect_url = f"{tr_endpoint.rstrip('/')}/detect?api-version=3.0"
    body = [{'Text': text}]
    detect_response = requests.post(detect_url, headers=headers, json=body)
    detect_response.raise_for_status()
    detected_lang = detect_response.json()[0]['language']
    detected_language_name = language_map.get(detected_lang, detected_lang)
    print(f"🕵️ Detected Language: {detected_lang} ({detected_language_name})")

    # Step 2: Translate to English
    translate_to_english_url = f"{tr_endpoint.rstrip('/')}/translate?api-version=3.0&from={detected_lang}&to=en"
    translate_response = requests.post(translate_to_english_url, headers=headers, json=body)
    translate_response.raise_for_status()
    english_text = translate_response.json()[0]['translations'][0]['text']
    # print("🔠 Translated to English:", english_text)
    translated = str(english_text) 
    return translated, detected_lang

async def translate_back_to_original(english_text, original_lang_code):
    body_back = [{'Text': english_text}]
    translate_back_url = f"{tr_endpoint.rstrip('/')}/translate?api-version=3.0&from=en&to={original_lang_code}"
    translate_back_response = requests.post(translate_back_url, headers=headers, json=body_back)
    translate_back_response.raise_for_status()
    back_translated_text = translate_back_response.json()[0]['translations'][0]['text']
    # print("🔁 Back-Translated to Original:", back_translated_text)
    return str(back_translated_text)

@cl.on_chat_start
async def initialize_chat():
    """Initializes the chat session with all agents"""
    global kernel, gmaps
    global legal_plugin, location_plugin
    
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="legal_agents",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    )
    
    # Initialize plugins
    legal_plugin = kernel.add_plugin(plugin_name="LegalAgents", parent_directory="plugins")
    location_plugin = kernel.add_plugin(plugin_name="LocationAgent", parent_directory="plugins")
    
    # Initialize session state
    cl.user_session.set("legal_plugin", legal_plugin)
    cl.user_session.set("location_plugin", location_plugin)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("session_start", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Initialize Google Maps client
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

    # Send initial message with actions
    await send_input_options()

async def send_input_options():
    """Helper function to send input options"""
    actions = [
        cl.Action(
            name="english_voice",
            value="en-IN",
            label="🗣️ English Voice",
            description="Speak in English",
            payload={"type": "voice", "language": "en-IN"}
        ),
        cl.Action(
            name="hindi_voice",
            value="hi-IN",
            label="🗣️ Hindi Voice (हिंदी)",
            description="हिंदी में बोलें",
            payload={"type": "voice", "language": "hi-IN"}
        ),

        cl.Action(
            name="text_input",
            value="text",
            label="✏️ Text Input",
            description="Type your message",
            payload={"type": "text"}
        ),
        cl.Action(
            name="marathi_voice",
            value="mr-IN",
            label="🗣️ Marathi Voice (मराठी)",
            description="मराठीत बोला",
            payload={"type": "voice", "language": "mr-IN"}
        ),
        cl.Action(
            name="bengali_voice",
            value="bn-IN",
            label="🗣️ Bengali Voice (বাংলা)",
            description="বাংলায় বলুন",
            payload={"type": "voice", "language": "bn-IN"}
        ),
        cl.Action(
            name="gujarati_voice",
            value="gu-IN",
            label="🗣️ Gujarati Voice (ગુજરાતી)",
            description="ગુજરાતીમાં બોલો",
            payload={"type": "voice", "language": "gu-IN"}
        ),
        cl.Action(
            name="kannada_voice",
            value="kn-IN",
            label="🗣️ Kannada Voice (ಕನ್ನಡ)",
            description="ಕನ್ನಡದಲ್ಲಿ ಮಾತಾಡಿ",
            payload={"type": "voice", "language": "kn-IN"}
        ),
        cl.Action(
            name="malayalam_voice",
            value="ml-IN",
            label="🗣️ Malayalam Voice (മലയാളം)",
            description="മലയാളത്തിൽ സംസാരിക്കുക",
            payload={"type": "voice", "language": "ml-IN"}
        ),
        cl.Action(
            name="punjabi_voice",
            value="pa-IN",
            label="🗣️ Punjabi Voice (ਪੰਜਾਬੀ)",
            description="ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲ ਕਰੋ",
            payload={"type": "voice", "language": "pa-IN"}
        ),
        cl.Action(
            name="tamil_voice",
            value="ta-IN",
            label="🗣️ Tamil Voice (தமிழ்)",
            description="தமிழில் பேசுங்கள்",
            payload={"type": "voice", "language": "ta-IN"}
        ),
        cl.Action(
            name="telugu_voice",
            value="te-IN",
            label="🗣️ Telugu Voice (తెలుగు)",
            description="తెలుగులో మాట్లాడండి",
            payload={"type": "voice", "language": "te-IN"}
        ),
        cl.Action(
            name="urdu_voice",
            value="ur-IN",
            label="🗣️ Urdu Voice (اردو)",
            description="اردو میں بات کریں",
            payload={"type": "voice", "language": "ur-IN"}
        )
    ]

    await cl.Message(content="How would you like to provide input?", actions=actions).send()

async def handle_voice_input(language_locale: str):
    """Common handler for voice input in any language"""
    try:
        listening_msg = await cl.Message(content=f"🎤 Listening in {language_locale[:2].upper()}... (Speak now)").send()
        transcript, detected_lang = await recognize_speech(language_locale)
        await listening_msg.remove()
        
        # Store detected language in session
        cl.user_session.set("detected_lang", detected_lang)
        
        # Create a message with the transcript and process it
        msg = cl.Message(content=transcript)
        await msg.send()
        
        # Process the message through the normal flow
        await handle_message(msg)
        
    except Exception as e:
        await cl.Message(content=f"❌ Voice input error: {str(e)}").send()

@cl.action_callback("english_voice")
async def on_english_voice(action: cl.Action):
    """Handle English voice input"""
    await handle_voice_input("en-IN")

@cl.action_callback("hindi_voice")
async def on_hindi_voice(action: cl.Action):
    """Handle Hindi voice input"""
    await handle_voice_input("hi-IN")

@cl.action_callback("marathi_voice")
async def on_marathi_voice(action: cl.Action):
    """Handle Marathi voice input"""
    await handle_voice_input("mr-IN")

@cl.action_callback("bengali_voice")
async def on_bengali_voice(action: cl.Action):
    """Handle Bengali voice input"""
    await handle_voice_input("bn-IN")

@cl.action_callback("gujarati_voice")
async def on_gujarati_voice(action: cl.Action):
    """Handle Gujarati voice input"""
    await handle_voice_input("gu-IN")

@cl.action_callback("kannada_voice")
async def on_kannada_voice(action: cl.Action):
    """Handle Kannada voice input"""
    await handle_voice_input("kn-IN")

@cl.action_callback("malayalam_voice")
async def on_malayalam_voice(action: cl.Action):
    """Handle Malayalam voice input"""
    await handle_voice_input("ml-IN")

@cl.action_callback("punjabi_voice")
async def on_punjabi_voice(action: cl.Action):
    """Handle Punjabi voice input"""
    await handle_voice_input("pa-IN")

@cl.action_callback("tamil_voice")
async def on_tamil_voice(action: cl.Action):
    """Handle Tamil voice input"""
    await handle_voice_input("ta-IN")

@cl.action_callback("telugu_voice")
async def on_telugu_voice(action: cl.Action):
    """Handle Telugu voice input"""
    await handle_voice_input("te-IN")

@cl.action_callback("urdu_voice")
async def on_urdu_voice(action: cl.Action):
    """Handle Urdu voice input"""
    await handle_voice_input("ur-IN")

@cl.action_callback("text_input")
async def on_text_input(action: cl.Action):
    """Handle text input selection"""
    await cl.Message(content="Please type your message in the chat").send()

@cl.action_callback("tts_command")
async def on_tts_command(action: cl.Action):
    """Handle TTS command for a message"""
    try:
        # Get the text from the action payload
        text_to_speak = action.payload["text"]
        detected_lang = cl.user_session.get("detected_lang", "en")
        
        status_msg = await cl.Message(content="🔊 Preparing speech...").send()
        success = await synthesize_speech(text_to_speak, detected_lang)
        await status_msg.remove()
        
        if not success:
            await cl.Message(content="⚠️ Could not generate speech for this message").send()
    except KeyError:
        await cl.Message(content="❌ Error: No text found to speak").send()
    except Exception as e:
        await cl.Message(content=f"❌ TTS Error: {str(e)}").send()

async def extract_city_from_message(message: str) -> Optional[str]:
    """Extracts standardized city name from current message"""
    try:
        location_plugin = cl.user_session.get("location_plugin")
        result = await kernel.invoke(
            location_plugin["city_validator"],
            arguments=KernelArguments(current_message=message)
        )
        city = str(result).strip()
        return city if city != "UNKNOWN" else None
    except Exception as e:
        print(f"City extraction error: {e}")
        return None

async def geocode_city(city: str) -> Optional[str]:
    """Converts city name to coordinates"""
    try:
        geocode_result = gmaps.geocode(f"{city}, India")
        if geocode_result:
            loc = geocode_result[0]['geometry']['location']
            await cl.Message(content=f"📍 Detected location: {city}").send()
            return f"{loc['lat']},{loc['lng']}"
    except Exception as e:
        print(f"Geocoding error for {city}: {e}")
    return None

async def detect_location_from_ip() -> Optional[str]:
    """Attempts location detection via IP address"""
    try:
        ip = cl.user_session.get("client").get("host")
        if ip not in ["127.0.0.1", "::1"]:
            response = requests.get(
                f"https://ipinfo.io/{ip}?token={os.getenv('IPINFO_TOKEN')}",
                timeout=3
            )
            if response.status_code == 200 and "loc" in response.json():
                data = response.json()
                await cl.Message(content=f"📍 IP detected location: {data.get('city', 'Unknown')}").send()
                return data["loc"]
    except Exception as e:
        print(f"IP detection error: {e}")
    return None

async def manual_location_fallback() -> str:
    """Handles manual city input with proper validation"""
    try:
        # Get user input with proper error handling
        res = await cl.AskUserMessage(
            content="📍 Please share your city (e.g. 'Mumbai'):",
            timeout=120
        ).send()

        if not res or not isinstance(res, dict) or 'output' not in res or not res['output'].strip():
            raise ValueError("No valid city input received")
        user_input = res['output'].strip()  # Corrected line
        print(f"DEBUG - User input: {user_input}")  # Log raw input

        # Get plugin and validate
        location_plugin = cl.user_session.get("location_plugin")
        if not location_plugin:
            raise ValueError("Location plugin not initialized")

        # Process through city validator
        validation_result = await kernel.invoke(
            location_plugin["city_validator"],
            arguments=KernelArguments(current_message=user_input)
        )

        clean_city = str(validation_result).strip()
        print(f"DEBUG - Validated city: {clean_city}")  # Log cleaned city
        if clean_city == "UNKNOWN":
            raise ValueError(f"Couldn't identify city from: {user_input}")

        # Geocode the city
        geocode_result = gmaps.geocode(f"{clean_city}, India")
        if not geocode_result:
            raise ValueError(f"Google Maps couldn't locate: {clean_city}")

        loc = geocode_result[0]['geometry']['location']
        await cl.Message(content=f"📍 Location set to: {clean_city}").send()
        return f"{loc['lat']},{loc['lng']}"

    except Exception as e:
        print(f"LOCATION ERROR: {str(e)}")
        await cl.Message(
            content=f"⚠️ Error: {str(e)}\nFalling back to Delhi"
        ).send()
        return "28.6139,77.2090"


async def get_user_location(message: str) -> str:
    """Optimized location detection flow"""
    try:
        # 1. Try extracting from current message
        if message:
            if city := await extract_city_from_message(message):
                if coords := await geocode_city(city):
                    return coords
    except Exception as e:
        print(f"Error extracting city from message: {e}")

    try:
        # 2. Try IP detection
        if coords := await detect_location_from_ip():
            return coords
    except Exception as e:
        print(f"IP detection failed: {e}")

    # 3. Manual fallback
    return await manual_location_fallback()

async def analyze_document(file: cl.File) -> str:
    try:
        progress_msg = await cl.Message(content="📄 Processing document...").send()
        content = ""
        file_path = file.path

        # Text files
        if file.name.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    return f"Text file reading error: {str(e)}"
        
        # PDF/Images (OCR)
        elif file.name.lower().endswith(('.pdf', '.jpg', '.png', '.tiff')):
            try:
                ocr_text = await ocr(file_path)
                content, _ = await detect_and_translate_to_english(ocr_text)  # Unpack tuple
            except Exception as e:
                return f"OCR processing error: {str(e)}"
        
        # Word documents
        elif file.name.endswith(('.docx', '.doc')):
            try:
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                return f"Word document error: {str(e)}"

        if not content.strip():
            return "Error: Document appears to be empty."
        
        # Update progress message
        progress_msg.content = "🔍 Analyzing document content..."
        await progress_msg.update()

        # Create arguments for semantic kernel
        arguments = KernelArguments(document_text=content[:4000])  # Limit to 4000 chars

        # Invoke document analyzer
        analysis = await kernel.invoke(
            plugin_name=legal_plugin.name,
            function_name="document_analyzer",
            arguments=arguments
        )

        # Get action recommendation
        progress_msg.content = "⚖️ Evaluating legal implications..."
        await progress_msg.update()

        action_required = await kernel.invoke(
            plugin_name=legal_plugin.name,
            function_name="action_required",
            arguments=KernelArguments(document_analysis=str(analysis))
        )

        # Format the response
        action_flag = str(action_required).strip().upper()
        action_text = ("⚠️ **Action Recommended** - Consult a lawyer immediately" 
                      if "YES" in action_flag 
                      else "✅ **No Immediate Action Needed**")

        # Create summary
        summary = await kernel.invoke(
            plugin_name=legal_plugin.name,
            function_name="document_summarizer",
            arguments=KernelArguments(document_text=content[:4000])
        )
        
        response = (
            f"## 📑 Document Analysis Summary\n"
            f"{str(summary).strip()}\n\n"
            f"## 🔍 Detailed Analysis\n"
            f"{str(analysis).strip()}\n\n"
            f"## ⚖️ Legal Assessment\n"
            f"{action_text}\n\n"
            f"📌 _Analysis based on {file.name}_"
        )
        
        return response
        
    except Exception as e:
        error_msg = (
            f"❌ Document analysis failed\n\n"
            f"**Error**: {str(e)}\n\n"
            f"Please try again or upload a different file."
        )
        return error_msg

async def get_lawyer_recommendations(coords: str, lawyer_type: str) -> str:
    """Finds nearby lawyers with ranking"""
    try:
        places = gmaps.places_nearby(
            location=coords,
            keyword=f"{lawyer_type} lawyer",
            radius=50000,
            type="lawyer",
        )
        
        top_lawyers = sorted(
            places.get("results", []),
            key=lambda x: (-x.get('rating', 0), x.get('user_ratings_total', 0)),
        )[:3]

        if not top_lawyers:
            return "No nearby lawyers found for this specialty"

        return "\n\n".join(
            f"🏛 **{lawyer['name']}** (⭐ {lawyer.get('rating', 'N/A')})\n"
            f"📍 {lawyer['vicinity']}\n"
            f"📞 Contact via Google Maps"
            for lawyer in top_lawyers
        )
    except Exception as e:
        return f"⚠️ Error finding lawyers: {str(e)}"

@cl.on_message
async def handle_message(message: cl.Message):
    # Get current chat history
    chat_history: List[Dict[str, Any]] = cl.user_session.get("chat_history", [])

    # Check if this is a voice input follow-up
    is_voice_input = message.content.startswith("🎤")  # Simple check for voice input
    detected_lang = cl.user_session.get("detected_lang", "en")

    # Keep only last 20 messages to prevent unlimited growth
    MAX_HISTORY = 20
    chat_history = chat_history[-MAX_HISTORY:]
    
    # Add user message to history
    chat_history.append({
        "role": "user",
        "content": message.content,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "input_mode": "voice" if is_voice_input else "text"
    })

    # Send input options after each message
    await send_input_options()
    
    # Handle file attachments
    if message.elements:
        processing_msg = await cl.Message(content="📄 Analyzing document...").send()
        analyses = []
        for element in message.elements:
            if isinstance(element, cl.File):
                analysis = await analyze_document(element)
                analyses.append(analysis)
        
        if analyses:
            # Add analysis to history
            chat_history.append({
                "role": "assistant",
                "content": "\n\n".join(analyses),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Create response message with TTS action
            response_content = "\n\n".join(analyses)
            actions = [
                cl.Action(
                    name="tts_command",
                    value=response_content,
                    label="🗣️ Speak Response",
                    description="Hear the response",
                    payload={"text": response_content}
                )
            ]
            await cl.Message(content=response_content, actions=actions).send()
            
            await processing_msg.remove()
            return
    
    # Process text query
    processing_msg = await cl.Message(content="🔍 Analyzing your query...").send()
    
    try:
        # Get the original text (remove voice prefix if present)
        original_text = message.content.replace("🎤 You said: ", "") if is_voice_input else message.content
        
        # Translate to English if needed
        translated_text, detected_lang = await detect_and_translate_to_english(original_text)
        cl.user_session.set("detected_lang", detected_lang)
        
        legal_plugin = cl.user_session.get("legal_plugin")
        
        # First determine if this is a legal query or casual chat
        query_type = await kernel.invoke(
            legal_plugin["query_classifier"],
            arguments=KernelArguments(user_query=translated_text)
        )
        query_type = str(query_type).strip().upper()

        if "CASUAL" in query_type:
            # Handle casual legal chat
            response = chat_client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You're a legal assistant. Provide helpful but non-binding responses to general legal questions."},
                    {"role": "user", "content": translated_text},
                ],
            )
            advice_text = response.choices[0].message.content
        
        else:
            # Get the last 3 messages for context
            context_messages = [
                msg["content"] for msg in chat_history[-3:] 
                if msg["role"] == "user" or msg["role"] == "assistant"
            ]
            context = str("\n".join(context_messages))

            selected_index = detect_relevant_index(translated_text)
            if not selected_index:
                print("No matching index found for query. Defaulting to index2.")
                selected_index = "index2"
            if selected_index == "index1":
                search_client = SearchClient(SEARCH_ENDPOINT, index_name_1, AzureKeyCredential(SEARCH_API_KEY))
            elif selected_index == "index2":
                search_client = SearchClient(SEARCH_ENDPOINT, index_name_2, AzureKeyCredential(SEARCH_API_KEY))
            else:
                search_client = SearchClient(SEARCH_ENDPOINT, index_name_3, AzureKeyCredential(SEARCH_API_KEY))

            embedding_data = []

            # Get query embedding first
            final_query = await embedder(translated_text + context)
            if not final_query:
                print("Query embedding failed.")
                return

            results_1 = search_client.search(
                search_text=translated_text,
                top=1,
                select=['Act_name', 'content', 'keyphrases']
            )

            for r in results_1:
                chunks = chunker(r['content'])
                for c in chunks:
                    embed = await embedder(c)
                    if embed:
                        embedding_data.append((embed, c, r['Act_name']))

            if not embedding_data:
                print("No valid embeddings retrieved from documents.")
                return

            # perform vector search
            best_match = vector_search(final_query, embedding_data)
            similarity_score, context_chunk, act_name = best_match

            needs_advice = await kernel.invoke(
                legal_plugin["needs_advice"],
                arguments=KernelArguments(
                    context=context_chunk,
                    question=translated_text
                )
            )
            needs_advice = str(needs_advice).strip().upper()

            if "YES" in needs_advice:
                # Generate formal legal advice
                response = chat_client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context: {context_chunk}\nQuestion: {translated_text}"},
                    ],
                )
                advice_text = response.choices[0].message.content
            else:
                # Return direct information
                advice_text = f"Relevant legal information from {act_name}:\n\n{context_chunk}"

            # Generate answer with OpenAI
            response = chat_client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is some context that will help you provide the answer: Extracted Content: {context_chunk}. The question is: {translated_text}"},
                ]
            )

            response = response.choices[0].message.content
        
            # Get location-aware legal advice with context
            user_location = await get_user_location(translated_text)
            advice_text = str(response).strip()
            trans_op = await translate_back_to_original(advice_text, detected_lang)
            
            # Add assistant response to history
            chat_history.append({
                "role": "assistant",
                "content": trans_op,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Check if lawyer is needed
            needs_lawyer = await kernel.invoke(
                legal_plugin["lawyer_needed"],
                arguments=KernelArguments(
                    legal_advice=str(advice_text),
                    user_query=translated_text,
                )
            )
            needs_lawyer = str(needs_lawyer).strip().upper()
            
            # Remove the processing message first
            await processing_msg.remove()
            
            # Create the response message
            if "YES" in needs_lawyer:
                lawyer_type = await kernel.invoke(
                    legal_plugin["lawyer_type"],
                    arguments=KernelArguments(legal_advice=str(advice_text))
                )
                lawyer_type = str(lawyer_type).strip()
                
                lawyers = await get_lawyer_recommendations(user_location, lawyer_type)
                final_content = f"## Legal Advice\n{trans_op}\n\n## Local {lawyer_type} Lawyers\n{lawyers}"
                
                # Update the final message in history
                chat_history[-1]["content"] = final_content
                
                # Create response message with TTS action
                actions = [
                    cl.Action(
                        name="tts_command",
                        value=final_content,
                        label="🗣️ Speak Response",
                        description="Hear the response",
                        payload={"text": final_content}
                    )
                ]
                await cl.Message(content=final_content, actions=actions).send()
                
                # Speak the response automatically if this was a voice input
                if is_voice_input:
                    try:
                        await synthesize_speech(trans_op, detected_lang)
                    except Exception as e:
                        await cl.Message(content=f"⚠️ Could not generate audio: {str(e)}").send()
                
            else:
                actions = [
                    cl.Action(
                        name="tts_command",
                        value=trans_op,
                        label="🗣️ Speak Response",
                        description="Hear the response",
                        payload={"text": trans_op}
                    )
                ]
                await cl.Message(content=f"✅ Advice:\n{trans_op}", actions=actions).send()
                
                # Speak the response automatically if this was a voice input
                if is_voice_input:
                    try:
                        await synthesize_speech(trans_op, detected_lang)
                    except Exception as e:
                        await cl.Message(content=f"⚠️ Could not generate audio: {str(e)}").send()
            
            # Update the chat history in session
            cl.user_session.set("chat_history", chat_history)
            
            # Remove processing message
            await processing_msg.remove()
                
    except Exception as e:
        error_msg = f"❌ Processing error: {str(e)}"
        chat_history.append({
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        cl.user_session.set("chat_history", chat_history)
        await processing_msg.remove()
        await cl.Message(content=error_msg).send()
