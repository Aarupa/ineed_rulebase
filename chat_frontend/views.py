#import speech_recognition as sr  # type: ignore 
#import pyttsx3  # type: ignore 
import json  
import random  
import spacy  # type: ignore
import nltk  # type: ignore
from nltk.corpus import wordnet # type: ignore
from textblob import TextBlob # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from fuzzywuzzy import process  # type: ignore
from django.http import JsonResponse # type: ignore
from django.views.decorators.csrf import csrf_exempt # type: ignore
from django.shortcuts import render # type: ignore
import os
from django.conf import settings # type: ignore
from django.template.loader import get_template # type: ignore
import logging
from django.http import HttpRequest # type: ignore
import time
import whisper  # Import Whisper
from pydub import AudioSegment
from pydub.playback import play
import tempfile
from gtts import gTTS

# ChatterBot imports
from chatterbot import ChatBot # type: ignore
from chatterbot.trainers import ChatterBotCorpusTrainer # type: ignore

import os
import json

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Define the path to content.json
json_path = os.path.join(script_dir, 'content.json')
dialogue_history_path = os.path.join(os.path.dirname(__file__), 'history.json')
# Load FAQ data
with open(json_path, 'r') as json_data:
    faq_data = json.load(json_data)

is_speaking = False
current_speech = None

#chatbot = ChatBot("MyBot")
#Create a trainer and train the chatbot
#trainer = ChatterBotCorpusTrainer(chatbot)

#Train chatbot with explicit corpus file paths
# trainer.train(
#     os.path.join(script_dir, "greetings.yml"),
#     os.path.join(script_dir, "conversations.yml")
# )


from chatterbot import ChatBot
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'db.sqlite3')

chatbot = ChatBot(
    "MyBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri=f"sqlite:///{model_path}"
)



nlp = spacy.load("en_core_web_sm")


nltk.download('wordnet')
sentiment_analyzer = SentimentIntensityAnalyzer()


conversation_history = []  # Store conversation history
history = {}

if os.path.exists(dialogue_history_path):
    with open(dialogue_history_path, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = {}
else:
    history = {}
'''
def save_conversation_to_file(user_message, response):
    """Append a new user-message and response pair to the JSON file."""
    history[user_message] = response
    with open(dialogue_history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
'''    

def save_conversation_to_file(user_message, response):
    """Save the conversation to a JSON file as key-value pairs."""
    # Update the history dictionary
    history[user_message] = response

    # Save the updated history to the JSON file
    with open(dialogue_history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def correct_spelling(query):
    """Correct spelling mistakes in the user query using TextBlob."""
    blob = TextBlob(query)
    corrected_query = str(blob.correct())  # Get the corrected version of the query
    return corrected_query

def get_best_match(query, choices, threshold=80, min_length=2):
    """Find the best matching FAQ keyword using fuzzy string matching."""
    # Ignore matching for very short words (likely typos, incomplete words, or greetings)
    if len(query) < min_length:
        return None  
    
    best_match, score = process.extractOne(query, choices)  

    # Ensure the match is strong and not just a substring of a larger keyword
    if score >= threshold:
        return best_match  

    return None    

def classify_query(msg):
    """Classify the query as company-related (FAQ) or general conversation."""
    msg_lower = msg.lower() 
    for i in  faq_data['faqs']:
        if msg_lower == i['question'].lower():
            response = random.choice(i['responses'])
            return "company", response
    else:
        # ----------------- FAQ Matching -----------------
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
                    response = random.choice(faq['responses'])
                    return "company", response

        # ----------------- General Conversation using ChatterBot -----------------
        response = chatbot.get_response(msg_lower)

        if response:
            return "general_convo", str(response)
        return "general", None  

def analyze_sentiment(msg):
    """Analyze the sentiment of a user message using VaderSentiment."""
    score = sentiment_analyzer.polarity_scores(msg)['compound'] 
    if score >= 0.5:
        return "positive"  
    elif score <= -0.5:
        return "negative" 
    return "neutral" 

def get_contextual_response(msg):
    """Check if the message relates to previous interactions and respond accordingly."""

    msg_lower = msg.lower().strip()
    
    # Avoid single words triggering full past responses
    if len(msg_lower.split()) < 2:
        return None  

    for prev_query, prev_response in reversed(conversation_history):
        if msg_lower in prev_query.lower():
            return random.choice(["As I mentioned earlier, ", "Like I said before, ", "Previously, I mentioned that "]) + prev_response
    return None

def generate_nlp_response(msg):
    """Generate a basic NLP response for general conversation."""
    doc = nlp(msg)  

    if any(token.lower_ in ["hi", "hello", "hey",'hii'] for token in doc):
        return random.choice(["Hey there! How's your day going?", "Hello! What’s up?", "Hi! How can I assist you today?"])
    
    elif "how are you" in msg.lower():
        return random.choice(["I'm doing great, thanks for asking! How about you?", "I'm good! Hope you're having a great day too."])
    
    elif msg.lower() in ["great","great!", "good","good!", "awesome","awesome!", "fantastic","fantastic!", "amazing", "amazing!"]:
        return random.choice(["Glad to hear that! 😊 What’s on your mind?", 
                              "That's awesome! How can I assist you today?", 
                              "Great! Let me know if you need any help."])
    
    
    elif "thank you" in msg.lower() or "thanks" in msg.lower():
        return random.choice(["You're very welcome!", "Anytime! Glad I could help."])
  
    elif msg.lower() in ["bye", "exit", "goodbye"]:
        conversation_history.clear() # Clear history
        return "Ok bye! Have a good day!"

    else:
        return "Could you clarify your question? I'm happy to help!"

@csrf_exempt
def get_response(request):
    """Handle HTTP requests and return chatbot responses as JSON."""
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('prompt', '')
        if user_message:
            # Clear history if user is ending the conversation
            if user_message.lower() in ["bye", "exit", "goodbye"]:
                conversation_history.clear()
                return JsonResponse({'text': "Ok bye! Have a good day!"})

            contextual_response = get_contextual_response(user_message)
            if contextual_response:
                # Update both the list and the JSON file
                conversation_history.append((user_message, contextual_response))
                save_conversation_to_file(user_message, contextual_response)
                return JsonResponse({'text': contextual_response})
            
            category, response = classify_query(user_message)
            if ((category == "company") or (category == "general_convo"))  and response:
                conversation_history.append((user_message, response)) # Store conversation history
                save_conversation_to_file(user_message, response)
                return JsonResponse({'text': response})
            
            
            response = generate_nlp_response(user_message)
            #conversation_history.append((user_message, response)) # Store conversation history
            return JsonResponse({'text': response})
    return JsonResponse({'text': 'Invalid request'}, status=400)

def speak(text):
    """Convert chatbot's text response to speech using gTTS."""
    try:
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en')
        
        # Save the speech to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            temp_audio_path = temp_audio_file.name

        # Play the audio using pydub
        audio = AudioSegment.from_file(temp_audio_path, format="mp3")
        play(audio)

        # Clean up the temporary file
        os.remove(temp_audio_path)

    except Exception as e:
        print(f"An error occurred during text-to-speech conversion: {e}")

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
    corrected_words = [corrections.get(word.lower(), word) for word in words]
    return " ".join(corrected_words)

# Load the Whisper model
whisper_model = whisper.load_model("tiny")

def listen():
    """Toggle microphone to listen continuously until user says 'bye' or 'exit'."""
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write

    mic_active = False
    print("Press Enter to toggle the microphone on/off. Say 'bye' or 'exit' to stop completely.")

    while True:
        command = input("Press Enter to toggle mic or type 'exit' to quit: ").strip().lower()

        if command == "exit" or command == "bye" or command == "bye bye":
            conversation_history.clear()  # Clear history
            print("Exiting the chat. Goodbye!")
            break

        mic_active = not mic_active

        if mic_active:
            print("Microphone is ON. Listening...")
            try:
                # Record audio
                duration = 15  # Record for 15 seconds
                print("Recording...")
                audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
                sd.wait()  # Wait until recording is finished
                print("Recording finished.")

                # Save audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    write(temp_audio_file.name, 16000, audio)
                    temp_audio_path = temp_audio_file.name

                # Convert WAV to MP3 using pydub and ffmpeg
                audio_segment = AudioSegment.from_wav(temp_audio_path)
                mp3_path = temp_audio_path.replace(".wav", ".mp3")
                audio_segment.export(mp3_path, format="mp3")
                print(f"Audio converted to MP3: {mp3_path}")

                # Transcribe audio using Whisper
                result = whisper_model.transcribe(temp_audio_path)
                user_message = result["text"]
                print(f"You said: {user_message}")

                # Clean up temporary files
                os.remove(temp_audio_path)
                os.remove(mp3_path)

                if "bye" in user_message.lower() or "exit" in user_message.lower() or "bye bye" in user_message.lower():
                    conversation_history.clear()  # Clear history
                    print("Exiting the chat. Goodbye!")
                    mic_active = False
                    break

                # Process the user message
                request = HttpRequest()
                request.method = 'POST'
                request.body = json.dumps({'prompt': user_message}).encode('utf-8')

                response = get_response(request)

                # Extract the text from the JsonResponse
                response_text = json.loads(response.content.decode('utf-8'))['text']
                print(f"Chatbot: {response_text}")

            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Microphone is OFF. Press Enter to toggle it back on.")

def chat(request):
    """Render the chatbot HTML template."""
    try:
        template_path = 'chatbot.html'
        logging.info(f"Attempting to load template: {template_path}")
        return render(request, template_path)
    except Exception as e:
        logging.error(f"Error loading template: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def clear_history(request):
    """Clear the conversation history."""
    global conversation_history
    if request.method == 'POST':
        conversation_history.clear()
        return JsonResponse({'status': 'success', 'message': 'Conversation history cleared.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)