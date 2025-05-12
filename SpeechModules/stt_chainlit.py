import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import chainlit as cl
import asyncio

# Load environment variables
load_dotenv()
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Validate Azure credentials
if not speech_key or not speech_region:
    raise RuntimeError("Missing Azure Speech credentials in .env file")

async def recognize_speech():
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Run the blocking .get() in a thread executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    future = recognizer.recognize_once_async()
    result = await loop.run_in_executor(None, future.get)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text.strip()
    elif result.reason == speechsdk.ResultReason.NoMatch:
        raise ValueError("No speech detected")
    else:
        raise ValueError(f"Recognition error: {result.cancellation_details.reason}")

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
                )
            ]
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

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
        await cl.Message(
            content="Try the voice button or type 'voice'",
            actions=[
                cl.Action(
                    name="voice_command",
                    value="voice",
                    label="🎤 Voice Input",
                    payload={"type": "voice"}
                )
            ]
        ).send()
