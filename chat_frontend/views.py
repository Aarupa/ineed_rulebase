# -------------------- Imports --------------------
# Standard library imports
import os
import json
import random
import tempfile
import logging
from datetime import datetime, timedelta
import string  # Ensure this import is present at the top of the file

# Third-party imports
from textblob import TextBlob  # Used in correct_spelling
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Used in analyze_sentiment
from fuzzywuzzy import process  # Used in get_best_match
from pydub import AudioSegment  # Used in speak and listen
from pydub.playback import play  # Used in speak
from gtts import gTTS  # Used in speak
import whisper  # Used in listen for transcribing audio
import requests

# Django imports
from django.http import JsonResponse, HttpRequest  # For views and simulated requests
from django.views.decorators.csrf import csrf_exempt  # For CSRF exemption on views
from django.shortcuts import render  # For rendering templates

import spacy  # Used in generate_nlp_response
import nltk  # Used to download 'wordnet'


# -------------------- Constants and Configurations --------------------
# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'json_files', 'content.json')
dialogue_history_path = os.path.join(script_dir, 'json_files', 'history.json')

# Initialize NLP and sentiment analysis tools
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')
sentiment_analyzer = SentimentIntensityAnalyzer()

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

# Clear history.json for one-time use
if os.path.exists(dialogue_history_path):
    with open(dialogue_history_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=4, ensure_ascii=False)



# Load Whisper model
whisper_model = whisper.load_model("tiny")

# Add chatbot's name
CHATBOT_NAME = "Infi"

# Load additional JSON files
greetings_path = os.path.join(script_dir, 'json_files', 'greetings.json')
farewells_path = os.path.join(script_dir, 'json_files', 'farewells.json')
general_path = os.path.join(script_dir, 'json_files', 'general.json')

with open(greetings_path, 'r', encoding='utf-8') as f:  # Specify encoding
    greetings_data = json.load(f)

with open(farewells_path, 'r', encoding='utf-8') as f:  # Specify encoding
    farewells_data = json.load(f)

with open(general_path, 'r', encoding='utf-8') as f:  # Specify encoding
    general_data = json.load(f)

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
        "indiaa": "indeed",
        "ended": "indeed",
        "inspiron": "inspiring",
        "inspire ring": "inspiring"
    }
    words = text.split()
    return " ".join(corrections.get(word.lower(), word) for word in words)

# -------------------- Chatbot Logic --------------------
def classify_query(msg):
    """Classify the query as company-related (FAQ) or general conversation."""
    # Normalize the message by removing punctuation and converting to lowercase
    msg_normalized = msg.translate(str.maketrans('', '', string.punctuation)).lower()

    # Handle "What is your name?" query
    if "what is your name" in msg_normalized or "your name" in msg_normalized:
        return "general_convo", f"My name is {CHATBOT_NAME}."

    greetings = [f"hey {CHATBOT_NAME.lower()}", f"hi {CHATBOT_NAME.lower()}","hi ","Hi ","hi  ","Hi  ","hello ","Hello ","hey ","Hey ","hii ","Hii ","hii  ","Hii  "]

    if any(greeting in msg_normalized for greeting in greetings):
        return "greeting", f"Hello! I'm {CHATBOT_NAME}. How can I assist you today?"

    for faq in faq_data['faqs']:
        if msg_normalized == faq['question'].translate(str.maketrans('', '', string.punctuation)).lower():
            return "company", random.choice(faq['responses'])

    keywords = [keyword.translate(str.maketrans('', '', string.punctuation)).lower() for faq in faq_data['faqs'] for keyword in faq['keywords']]
    best_match = get_best_match(msg_normalized, keywords)

    corrected_msg = correct_spelling(msg_normalized)
    if corrected_msg != msg_normalized:
        for faq in faq_data['faqs']:
            if best_match in faq['keywords']:
                suggested_question = faq['question']
                response = random.choice(faq['responses'])
                return "company", f"Did you mean '{suggested_question}'?\n {response}"

    if best_match:
        for faq in faq_data['faqs']:
            if best_match in faq['keywords']:
                return "company", random.choice(faq['responses'])

    # If no match is found, return a default response
    return None, None

def generate_nlp_response(msg):
    """Generate a basic NLP response for general conversation."""


    doc = nlp(msg)
    if any(token.lower_ in ["hi ", "hello", "hey", "hii"] for token in doc):
        return random.choice(["Hey there! How's your day going?", "Hello! What’s up?", "Hi! How can I assist you today?"])
    elif "how are you" in msg.lower():
        return random.choice(["I'm doing great, thanks for asking! How about you?", "I'm good! Hope you're having a great day too."])
    elif msg.lower() in ["great", "good", "awesome", "fantastic", "amazing"]:
        return random.choice(["Glad to hear that! 😊 What’s on your mind?", "That's awesome! How can I assist you today?"])
    elif "thank you" in msg.lower() or "thanks" in msg.lower():
        return random.choice(["You're very welcome!", "Anytime! Glad I could help."])
    elif msg.lower() in ["bye", "exit"]:
        conversation_history.clear()
        return "Ok bye! Have a good day!"
    else:
        return None

def get_contextual_response(user_message):
    """Generate a contextual response based on the user's message."""
    # Example logic for generating a contextual response
    if "weather" in user_message.lower():
        return "I'm not equipped to provide weather updates, but you can check a weather app!"
    elif "time" in user_message.lower():
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    return None

def handle_time_based_greeting(msg):
    """Handle time-based greetings and provide an appropriate response."""
    greetings = ["good morning", "good afternoon", "good evening", "good night"]
    msg_lower = msg.lower()

    # Check if the message contains a time-based greeting
    for greeting in greetings:
        if greeting in msg_lower:
            current_hour = datetime.now().hour
            if greeting == "good morning":
                if current_hour < 12:
                    return "Good morning! How can I assist you today?"
                elif current_hour < 18:
                    return "It's already afternoon, but good day to you!"
                else:
                    return "It's evening now, but good day to you!"
            elif greeting == "good afternoon":
                if current_hour < 12:
                    return "It's still morning, but good day to you!"
                elif current_hour < 18:
                    return "Good afternoon! How can I assist you today?"
                else:
                    return "It's evening now, but good day to you!"
            elif greeting == "good evening":
                if current_hour < 12:
                    return "It's still morning, but good day to you!"
                elif current_hour < 18:
                    return "It's still afternoon, but good day to you!"
                else:
                    return "Good evening! How can I assist you today?"
            elif greeting == "good night":
                return "Good night! Sleep well and take care!"

    # Handle queries about the current time
    if "current time" in msg_lower or "current time" in msg_lower:
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    # Fallback to classify query or ChatterBot response
    category, response = classify_query(msg)
    if response:
        return response

    
def handle_date_related_queries(msg):
    """Handle date-related queries and provide an appropriate response."""
    msg_lower = msg.lower()
    today = datetime.now()
    
    # Define a mapping for generic conditions
    date_mapping = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "day after tomorrow": today + timedelta(days=2),
        "yesterday": today - timedelta(days=1),
        "day before yesterday": today - timedelta(days=2),
        "next week": today + timedelta(weeks=1),
        "last week": today - timedelta(weeks=1),
        "next month": (today.replace(day=28) + timedelta(days=4)).replace(day=1),  # First day of next month
        "last month": (today.replace(day=1) - timedelta(days=1)).replace(day=1),  # First day of last month
        "next year": today.replace(year=today.year + 1),
        "last year": today.replace(year=today.year - 1)
    }
    
    # Check for specific phrases in the message
    for key, date in date_mapping.items():
        if key in msg_lower:
            if "date" in msg_lower:
                return f"The {key}'s date is {date.strftime('%B %d, %Y')}."
            elif "day" in msg_lower:
                return f"The {key} is {date.strftime('%A')}."
    
    # Fallback for unrecognized queries
    return None

def get_priority_response(preprocessed_input):
    """
    Check if the input matches greetings, farewells, or general responses.
    """
    # Normalize input by removing punctuation and converting to lowercase
    normalized_input = preprocessed_input.translate(str.maketrans('', '', string.punctuation)).lower()
    for category, data in [("greetings", greetings_data["greetings"]),
                           ("farewells", farewells_data["farewells"]),
                           ("general", general_data["general"])]:
        if normalized_input in map(str.lower, data["inputs"]):  # Case-insensitive matching
            logging.debug(f"Matched {category} for input: {normalized_input}")
            return random.choice(data["responses"])
    logging.debug(f"No match found in priority responses for input: {normalized_input}")
    return None

# ----------------------------------------

model = "mistralai/mistral-7b-instruct"

API_KEY = "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"

def get_response_openrouter(prompt):
    """Send request to OpenRouter with restrictions to general conversations only"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are a simple conversation bot that only handles very general conversations.
    Your responses should be limited to:
    - Greetings (e.g., hello, hi, good morning/afternoon/evening)
    - Farewells (e.g., goodbye, see you later, have a nice day)
    - Simple positive/negative responses (e.g., "That's nice!", "I'm sorry to hear that")
    - Basic acknowledgments (e.g., "I understand", "Okay")
    
    If asked about anything else, respond with: "I only handle general conversations like greetings and simple responses."
    """
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def is_general_conversation(text):
    """Check if the input text is a general conversation topic."""
    text = text.lower().strip()

    # Expanded lists of general conversation topics
    greetings = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy",
        "yo", "what's up", "hi there", "hello there", "hey there", "sup", "hiya", "morning", "evening",
        "how's it going", "how's everything", "what's new", "how do you do", "salutations", "good day",
        "how are you doing", "what's happening", "how's your day", "how's life", "hello world"
    ]
    farewells = [
        "bye", "goodbye", "see you", "farewell", "have a good day", "see ya", "take care", "later",
        "catch you later", "talk to you later", "bye bye", "peace", "good night", "see you soon",
        "see you later", "adios", "ciao", "so long", "until next time", "goodbye for now", "take it easy",
        "have a great day", "stay safe", "see you around", "cheerio", "all the best"
    ]
    positives = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "great", "awesome", "fine", "absolutely", "definitely",
        "of course", "sure thing", "yup", "cool", "sounds good", "alright", "I'm good", "perfect", "no problem",
        "that's nice", "wonderful", "amazing", "fantastic", "excellent", "terrific", "splendid", "superb",
        "brilliant", "outstanding", "marvelous", "lovely", "fabulous", "phenomenal", "incredible", "exceptional"
    ]
    negatives = [
        "no", "nope", "nah", "not really", "not good", "never", "no way", "I don't think so", "not at all",
        "unfortunately not", "doesn't work", "not happening", "no thanks", "bad", "terrible", "horrible",
        "awful", "dreadful", "poor", "unacceptable", "disappointing", "unsatisfactory", "subpar", "lousy",
        "miserable", "pathetic", "tragic", "unfortunate", "not great", "not okay", "not fine", "not ideal"
    ]
    common_phrases = [
        "how are you", "what's up", "what's your name", "thank you", "thanks", "you're welcome", "nice to meet you",
        "good to see you", "how's it going", "how do you do", "what's new", "how's life", "how's everything",
        "pleased to meet you", "it's a pleasure", "how can I help you", "how can I assist you", "what can I do for you",
        "long time no see", "it's been a while", "what's going on", "what's the matter", "how have you been",
        "what's your favorite", "tell me more", "can you help me", "I need assistance", "thank you so much",
        "much appreciated", "you're the best", "no worries", "don't mention it", "anytime","I am good", "I am fine",
        "I am okay", "I am doing well", "I am alright", "I am doing fine", "I am doing great", "I am doing good",
    ]

    # Combine all phrases into a single list for matching
    general_conversation_phrases = greetings + farewells + positives + negatives + common_phrases

    # Check for exact matches or partial matches
    for phrase in general_conversation_phrases:
        if phrase in text:
            return True

    # Fallback to fuzzy matching for more flexibility
    from fuzzywuzzy import fuzz
    for phrase in general_conversation_phrases:
        if fuzz.partial_ratio(phrase, text) > 80:  # Adjust threshold as needed
            return True

    return False

# -------------------- HTTP Request Handlers --------------------
@csrf_exempt
def get_response(request):
    """Handle HTTP requests and return chatbot responses as JSON."""
    global conversation_history
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('prompt', '')
        logging.debug(f"User message: {user_message}")  # Log the user message
        if user_message:
            # Preprocess the user message
            preprocessed_message = user_message.strip().lower()
            logging.debug(f"Preprocessed message: {preprocessed_message}")  # Log the preprocessed message

            # Handle date-related queries first
            date_related_response = handle_date_related_queries(preprocessed_message)
            if date_related_response:
                conversation_history.append((user_message, date_related_response))
                save_conversation_to_file(user_message, date_related_response)
                return JsonResponse({'text': date_related_response})

            # Handle time-based greetings
            time_based_response = handle_time_based_greeting(preprocessed_message)
            if time_based_response:
                conversation_history.append((user_message, time_based_response))
                save_conversation_to_file(user_message, time_based_response)
                return JsonResponse({'text': time_based_response})

            # Handle general NLP responses (e.g., "Hi", "Hello")
            nlp_response = generate_nlp_response(preprocessed_message)
            if nlp_response:
                conversation_history.append((user_message, nlp_response))
                save_conversation_to_file(user_message, nlp_response)
                return JsonResponse({'text': nlp_response})

            # Handle contextual responses
            contextual_response = get_contextual_response(preprocessed_message)
            if contextual_response:
                conversation_history.append((user_message, contextual_response))
                save_conversation_to_file(user_message, contextual_response)
                return JsonResponse({'text': contextual_response})

            # Handle classified queries
            category, response = classify_query(preprocessed_message)
            if response:
                conversation_history.append((user_message, response))
                save_conversation_to_file(user_message, response)
                return JsonResponse({'text': response})

            # Handle priority responses (greetings, farewells, general responses)
            priority_response = get_priority_response(preprocessed_message)
            if priority_response:
                conversation_history.append((user_message, priority_response))
                save_conversation_to_file(user_message, priority_response)
                return JsonResponse({'text': priority_response})

            # Use LLM only for small talk (general conversation topics)
            if is_general_conversation(preprocessed_message):
                llm_response = get_response_openrouter(preprocessed_message)
                conversation_history.append((user_message, llm_response))
                save_conversation_to_file(user_message, llm_response)
                return JsonResponse({'text': llm_response})

            # If no match is found, return a default response
            default_response = "I'm sorry, I don't have information on that topic."
            conversation_history.append((user_message, default_response))
            save_conversation_to_file(user_message, default_response)
            return JsonResponse({'text': default_response})
        
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