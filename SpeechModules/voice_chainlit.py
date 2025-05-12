import os
import asyncio
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import azure.cognitiveservices.speech as speechsdk
import chainlit as cl

# Load environment variables
load_dotenv()
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

if not speech_key or not speech_region:
    raise RuntimeError("Missing Azure Speech credentials in .env file")

# Map language name to locale code
indian_languages = {
    "hindi": "hi-IN",
    "marathi": "mr-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "urdu": "ur-IN",
    "english": "en-IN"
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
    'ur': 'ur-IN-SalmanNeural',
}

def get_speech_config():
    return speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)

async def recognize_speech(language_locale="en-IN"):
    speech_config = get_speech_config()
    speech_config.speech_recognition_language = language_locale
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    try:
        loop = asyncio.get_running_loop()
        future = recognizer.recognize_once_async()
        result = await loop.run_in_executor(None, future.get)

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip()
        elif result.reason == speechsdk.ResultReason.NoMatch:
            raise ValueError("No speech detected")
        else:
            raise ValueError(f"Recognition error: {result.cancellation_details.reason}")
    except Exception as e:
        raise ValueError(f"Speech recognition failed: {str(e)}")

def synthesize_speech(text):
    try:
        lang_code = detect(text)
        voice_name = language_to_voice.get(lang_code, 'en-IN-NeerjaNeural')
        
        speech_config = get_speech_config()
        speech_config.speech_synthesis_voice_name = voice_name
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return True, f"Speech synthesized in [{lang_code}] using voice [{voice_name}]"
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(f"TTS canceled: {details.reason} - {details.error_details}")
    except LangDetectException:
        # Fallback to English if language detection fails
        voice_name = 'en-IN-NeerjaNeural'
        speech_config = get_speech_config()
        speech_config.speech_synthesis_voice_name = voice_name
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()
        return True, f"Speech synthesized in [en] using voice [{voice_name}] (fallback)"
    except Exception as e:
        raise RuntimeError(f"TTS error: {str(e)}")

async def handle_response(transcript, locale):
    await cl.Message(
        content=f"**You said ({locale}):** {transcript}",
        actions=[
            cl.Action(
                name="voice_command",
                value="voice",
                label="🎤 Speak Again",
                payload={"type": "voice"}
            ),
            cl.Action(
                name="tts_command",
                value=transcript,
                label="🗣️ Speak this",
                payload={"type": "tts", "text": transcript}
            )
        ]
    ).send()

@cl.action_callback("voice_command")
async def on_voice_action(action: cl.Action):
    try:
        listening_msg = await cl.Message(content="🔊 Listening... (Speak now)").send()
        transcript = await recognize_speech()
        await listening_msg.remove()
        await handle_response(transcript, "auto-detected")
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.action_callback("tts_command")
async def on_tts_action(action: cl.Action):
    text = action.payload.get("text", "")
    if not text:
        await cl.Message(content="No text to speak.").send()
        return
    try:
        success, message = synthesize_speech(text)
        await cl.Message(content=f"✅ {message}").send()
    except Exception as e:
        await cl.Message(content=f"❌ TTS Error: {str(e)}").send()

@cl.action_callback("text_input")
async def on_text_action(action: cl.Action):
    await cl.Message(content="Please type your message in the chat").send()

@cl.action_callback("language_selection")
async def on_language_selection(action: cl.Action):
    locale = action.payload.get("locale", "en-IN")  # Changed from action.value to action.payload.get()
    lang_name = next((k for k, v in indian_languages.items() if v == locale), "English")
    await cl.Message(content=f"Selected language: {lang_name.capitalize()}").send()
    
    try:
        listening_msg = await cl.Message(content="🔊 Listening...").send()
        transcript = await recognize_speech(language_locale=locale)
        await listening_msg.remove()
        await handle_response(transcript, locale)
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.on_chat_start
async def init():
    actions = [
        cl.Action(
            name="language_selection",
            value=lang.capitalize(),
            label=f"🗣️ {lang.capitalize()}",
            description=f"Use {lang.capitalize()} voice",
            payload={"locale": locale}  # Store the locale in payload
        ) for lang, locale in indian_languages.items()
    ]
    actions.append(
        cl.Action(
            name="text_input",
            value="text",
            label="✏️ Text Input",
            description="Type your message instead",
            payload={"type": "text"}
        )
    )
    
    await cl.Message(
        content="Choose your input method or language:",
        actions=actions
    ).send()

@cl.on_message
async def main(message: cl.Message):
    if message.content.lower() == "voice":
        await on_voice_action(cl.Action(name="voice_command", value="voice", payload={"type": "voice"}))
    else:
        await cl.Message(
            content=f"You typed: {message.content}",
            actions=[
                cl.Action(
                    name="tts_command",
                    value=message.content,
                    label="🗣️ Speak this",
                    payload={"type": "tts", "text": message.content}
                ),
                cl.Action(
                    name="voice_command",
                    value="voice",
                    label="🎤 Voice Input",
                    payload={"type": "voice"}
                )
            ]
        ).send()