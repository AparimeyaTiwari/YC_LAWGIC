import os
import asyncio
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import chainlit as cl

# Load environment variables
load_dotenv()
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

if not speech_key or not speech_region:
    raise RuntimeError("Missing Azure Speech credentials in .env file")

# Azure Speech Config (shared)
def get_speech_config():
    config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    return config

# Speech-to-Text (STT)
async def recognize_speech():
    speech_config = get_speech_config()
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    loop = asyncio.get_running_loop()
    future = recognizer.recognize_once_async()
    result = await loop.run_in_executor(None, future.get)
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text.strip()
    elif result.reason == speechsdk.ResultReason.NoMatch:
        raise ValueError("No speech detected")
    else:
        raise ValueError(f"Recognition error: {result.cancellation_details.reason}")

# Text-to-Speech (TTS)
def synthesize_speech(text, voice="en-IN-NeerjaNeural"):
    speech_config = get_speech_config()
    speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return True
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"TTS canceled: {details.reason} - {details.error_details}")

# Chainlit Actions

@cl.action_callback("voice_command")
async def on_voice_action(action: cl.Action):
    try:
        listening_msg = await cl.Message(content="🔊 Listening... (Speak now)").send()
        transcript = await recognize_speech()
        await listening_msg.remove()
        await cl.Message(
            content=f"**You said:** {transcript}",
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
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

@cl.action_callback("tts_command")
async def on_tts_action(action: cl.Action):
    text = action.payload.get("text", "")
    if not text:
        await cl.Message(content="No text to speak.").send()
        return
    try:
        synthesize_speech(text)
        await cl.Message(content="✅ Spoke the text above!").send()
    except Exception as e:
        await cl.Message(content=f"❌ TTS Error: {str(e)}").send()


@cl.action_callback("text_input")
async def on_text_action(action: cl.Action):
    await cl.Message(content="Please type your message in the chat").send()

@cl.on_chat_start
async def init():
    actions = [
        cl.Action(
            name="voice_command",
            value="voice",
            label="🎤 Voice Input",
            payload={"type": "voice"}
        ),
        cl.Action(
            name="text_input",
            value="text",
            label="✏️ Text Input",
            payload={"type": "text"}
        )
    ]
    await cl.Message(
        content="How would you like to proceed?",
        actions=actions
    ).send()

@cl.on_message
async def main(message: cl.Message):
    if message.content.lower() == "voice":
        await on_voice_action(cl.Action(value="voice"))
    else:
        # Optionally, speak the typed text
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
