import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load environment variables from .env
load_dotenv()
speech_key = os.getenv("SPEECH_KEY")
speech_region = os.getenv("SPEECH_REGION")

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

def recognize_once(language_locale):
    # Configure speech
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = language_locale

    # Optional: extend silence timeout
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print(f"Listening... Speak something in {language_locale}.")

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized:", result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("Canceled:", cancellation.reason)
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print("Error details:", cancellation.error_details)

if __name__ == "__main__":
    print("Choose a language to speak from this list:")
    for lang in indian_languages:
        print("-", lang)

    chosen = input("\nEnter language: ").strip().lower()

    if chosen in indian_languages:
        recognize_once(indian_languages[chosen])
    else:
        print("Invalid language selection.")
