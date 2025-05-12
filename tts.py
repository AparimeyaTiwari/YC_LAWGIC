import os
from dotenv import load_dotenv
from langdetect import detect
import azure.cognitiveservices.speech as speechsdk

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

# Mapping from language code to Azure voice names
language_to_voice = {
    'en': 'en-IN-NeerjaNeural',       # English (India)
    'hi': 'hi-IN-SwaraNeural',        # Hindi
    'bn': 'bn-IN-TanishaaNeural',     # Bengali
    'gu': 'gu-IN-DhwaniNeural',       # Gujarati
    'kn': 'kn-IN-SapnaNeural',        # Kannada
    'ml': 'ml-IN-SobhanaNeural',      # Malayalam
    'mr': 'mr-IN-AarohiNeural',       # Marathi
    'pa': 'pa-IN-GaganNeural',        # Punjabi
    'ta': 'ta-IN-PallaviNeural',      # Tamil
    'te': 'te-IN-MohanNeural',        # Telugu
    'ur': 'ur-IN-SalmanNeural',       # Urdu
}

# Get text input
print("Enter some text that you want to speak >")
text = input()

# Detect language
lang_code = detect(text)
print(f"Detected language: {lang_code}")

# Pick voice
voice_name = language_to_voice.get(lang_code, 'en-IN-NeerjaNeural')

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_synthesis_voice_name = voice_name
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print(f"Speech synthesized in [{lang_code}] using voice [{voice_name}] for text: {text}")
elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_synthesis_result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        if cancellation_details.error_details:
            print("Error details: {}".format(cancellation_details.error_details))
