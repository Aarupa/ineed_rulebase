# -------------------- Imports --------------------
# Standard library imports
import os
import json
import random
import tempfile
import logging

# Third-party imports
import speech_recognition as sr  # type: ignore
import pyttsx3  # type: ignore
import spacy  # type: ignore
import nltk  # type: ignore
from nltk.corpus import wordnet  # type: ignore
from textblob import TextBlob  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from fuzzywuzzy import process  # type: ignore
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import whisper  # Import Whisper

# Django imports
from django.http import JsonResponse, HttpRequest  # type: ignore
from django.views.decorators.csrf import csrf_exempt  # type: ignore
from django.shortcuts import render  # type: ignore
from django.conf import settings  # type: ignore
from django.template.loader import get_template  # type: ignore

# ChatterBot imports
from chatterbot import ChatBot  # type: ignore
from chatterbot.trainers import ChatterBotCorpusTrainer  # type: ignore

# -------------------- Constants and Configurations --------------------
# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'content.json')
dialogue_history_path = os.path.join(script_dir, 'history.json')
model_path = os.path.join(script_dir, 'db.sqlite3')

# Initialize NLP and sentiment analysis tools
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')
sentiment_analyzer = SentimentIntensityAnalyzer()
engine = pyttsx3.init()

# Initialize ChatterBot
chatbot = ChatBot(
    "MyBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri=f"sqlite:///{model_path}"
)

# Load FAQ data
with open(json_path, 'r') as json_data:
    faq_data = json.load(json_data)

# Conversation history
conversation_history = []
history = {}
if os.path.exists(dialogue_history_path):
    with open(dialogue_history_path, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = {}

# Load Whisper model
whisper_model = whisper.load_model("tiny")

# -------------------- Utility Functions --------------------
def save_conversation_to_file(user_message, response):
    """Save the conversation to a JSON file as key-value pairs."""
    history[user_message] = response
    with open(dialogue_history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def correct_spelling(query):
    """Correct spelling mistakes in the user query using TextBlob."""
    blob = TextBlob(query)
    return str(blob.correct())

def get_best_match(query, choices, threshold=80, min_length=2):
    """Find the best matching FAQ keyword using fuzzy string matching."""
    if len(query) < min_length:
        return None
    best_match, score = process.extractOne(query, choices)
    return best_match if score >= threshold else None

def analyze_sentiment(msg):
    """Analyze the sentiment of a user message using VaderSentiment."""
    score = sentiment_analyzer.polarity_scores(msg)['compound']
    if score >= 0.5:
        return "positive"
    elif score <= -0.5:
        return "negative"
    return "neutral"

def preprocess_recognized_text(text):
    """Correct common misinterpretations in recognized speech."""
    corrections = {
        "crushal": "prushal",
        "india": "indeed",
        "ended": "indeed",
        "inspiron": "inspiring",
        "inspire ring": "inspiring"
    }
    words = text.split()
    return " ".join(corrections.get(word.lower(), word) for word in words)

# -------------------- Chatbot Logic --------------------
def classify_query(msg):
    """Classify the query as company-related (FAQ) or general conversation."""
    msg_lower = msg.lower()
    if msg_lower in ["hi", "hello", "hey", "hii"]:
        return None, None

    for faq in faq_data['faqs']:
        if msg_lower == faq['question'].lower():
            return "company", random.choice(faq['responses'])

    keywords = [keyword for faq in faq_data['faqs'] for keyword in faq['keywords']]
    best_match = get_best_match(msg_lower, keywords)

    corrected_msg = correct_spelling(msg_lower)
    if corrected_msg != msg_lower:
        for faq in faq_data['faqs']:
            if best_match in faq['keywords']:
                suggested_question = faq['question']
                response = random.choice(faq['responses'])
                return "company", f"Did you mean '{suggested_question}'?\n {response}"

    if best_match:
        for faq in faq_data['faqs']:
            if best_match in faq['keywords']:
                return "company", random.choice(faq['responses'])

    response = chatbot.get_response(msg_lower)
    return "general_convo", str(response) if response else None

def generate_nlp_response(msg):
    """Generate a basic NLP response for general conversation."""
    doc = nlp(msg)
    if any(token.lower_ in ["hi", "hello", "hey", "hii"] for token in doc):
        return random.choice(["Hey there! How's your day going?", "Hello! What’s up?", "Hi! How can I assist you today?"])
    elif "how are you" in msg.lower():
        return random.choice(["I'm doing great, thanks for asking! How about you?", "I'm good! Hope you're having a great day too."])
    elif msg.lower() in ["great", "good", "awesome", "fantastic", "amazing"]:
        return random.choice(["Glad to hear that! 😊 What’s on your mind?", "That's awesome! How can I assist you today?"])
    elif "thank you" in msg.lower() or "thanks" in msg.lower():
        return random.choice(["You're very welcome!", "Anytime! Glad I could help."])
    elif msg.lower() in ["bye", "exit", "goodbye"]:
        conversation_history.clear()
        return "Ok bye! Have a good day!"
    else:
        return "Could you clarify your question? I'm happy to help!"

def get_contextual_response(user_message):
    """Generate a contextual response based on the user's message."""
    # Example logic for generating a contextual response
    if "weather" in user_message.lower():
        return "I'm not equipped to provide weather updates, but you can check a weather app!"
    elif "time" in user_message.lower():
        from datetime import datetime
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    return None

# -------------------- HTTP Request Handlers --------------------
@csrf_exempt
def get_response(request):
    """Handle HTTP requests and return chatbot responses as JSON."""
    global conversation_history
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('prompt', '')
        if user_message:
            if user_message.lower() in ["bye", "exit", "goodbye", "bye bye"]:
                conversation_history.clear()
                return JsonResponse({'text': "Ok bye! Have a good day!"})

            contextual_response = get_contextual_response(user_message)
            if contextual_response:
                conversation_history.append((user_message, contextual_response))
                save_conversation_to_file(user_message, contextual_response)
                return JsonResponse({'text': contextual_response})

            category, response = classify_query(user_message)
            if response:
                conversation_history.append((user_message, response))
                save_conversation_to_file(user_message, response)
                return JsonResponse({'text': response})

            response = generate_nlp_response(user_message)
            return JsonResponse({'text': response})
    return JsonResponse({'text': 'Invalid request'}, status=400)

@csrf_exempt
def clear_history(request):
    """Clear the conversation history."""
    global conversation_history
    if request.method == 'POST':
        conversation_history.clear()
        return JsonResponse({'status': 'success', 'message': 'Conversation history cleared.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

def chat(request):
    """Render the chatbot HTML template."""
    try:
        return render(request, 'chatbot.html')
    except Exception as e:
        logging.error(f"Error loading template: {e}")
        return JsonResponse({'error': str(e)}, status=500)

# -------------------- Speech and Listening Functions --------------------
def speak(text):
    """Convert chatbot's text response to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            temp_audio_path = temp_audio_file.name
        audio = AudioSegment.from_file(temp_audio_path, format="mp3")
        play(audio)
        os.remove(temp_audio_path)
    except Exception as e:
        print(f"An error occurred during text-to-speech conversion: {e}")

def listen():
    """Toggle microphone to listen continuously until user says 'bye' or 'exit'."""
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write

    mic_active = False
    print("Press Enter to toggle the microphone on/off. Say 'bye' or 'exit' to stop completely.")

    while True:
        command = input("Press Enter to toggle mic or type 'exit' to quit: ").strip().lower()
        if command in ["exit", "bye", "bye bye"]:
            conversation_history.clear()
            print("Exiting the chat. Goodbye!")
            break

        mic_active = not mic_active
        if mic_active:
            print("Microphone is ON. Listening...")
            try:
                duration = 15
                audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
                sd.wait()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    write(temp_audio_file.name, 16000, audio)
                    temp_audio_path = temp_audio_file.name
                audio_segment = AudioSegment.from_wav(temp_audio_path)
                mp3_path = temp_audio_path.replace(".wav", ".mp3")
                audio_segment.export(mp3_path, format="mp3")
                result = whisper_model.transcribe(temp_audio_path)
                user_message = result["text"]
                os.remove(temp_audio_path)
                os.remove(mp3_path)
                if "bye" in user_message.lower() or "exit" in user_message.lower():
                    conversation_history.clear()
                    print("Exiting the chat. Goodbye!")
                    mic_active = False
                    break
                request = HttpRequest()
                request.method = 'POST'
                request.body = json.dumps({'prompt': user_message}).encode('utf-8')
                response = get_response(request)
                response_text = json.loads(response.content.decode('utf-8'))['text']
                print(f"Chatbot: {response_text}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Microphone is OFF. Press Enter to toggle it back on.")