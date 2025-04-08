import speech_recognition as sr  # type: ignore 
import pyttsx3  # type: ignore 
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
from textblob import TextBlob # type: ignore
from datetime import datetime
import time

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

datafest_path = os.path.join(script_dir, 'datafest.json')
with open(datafest_path, 'r') as file:
        datafest = json.load(file)

# Load DataFest FAQ data
datafest_faq_path = os.path.join(script_dir, 'datafest.json')
with open(datafest_faq_path, 'r') as file:
    datafest_faq_data = json.load(file)

# Combine FAQ data from both files
combined_faq_data = faq_data['faqs'] + datafest_faq_data['faqs']

# Use the same directory as content.json for corpus files
corpus_path = script_dir  
is_speaking = False
current_speech = None

chatbot = ChatBot("MyBot")
# Create a trainer and train the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train chatbot with explicit corpus file paths
trainer.train(
    os.path.join(script_dir, "greetings.yml"),
    os.path.join(script_dir, "conversations.yml")
)


nlp = spacy.load("en_core_web_sm")


nltk.download('wordnet')
sentiment_analyzer = SentimentIntensityAnalyzer()
engine = pyttsx3.init()

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

    keywords = [keyword for faq in combined_faq_data for keyword in faq['keywords']]
    best_match = get_best_match(msg_lower, keywords)
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
        for faq in combined_faq_data:
            if best_match in faq['keywords']:
                suggested_question = faq['question']
                response = random.choice(faq['responses'])
                return "company", f"Did you mean '{suggested_question}'?\n {response}"

    if best_match:
        for faq in combined_faq_data:
            if best_match in faq['keywords']:
                response = random.choice(faq['responses'])
                return "company", response
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

def get_event_status():
    """Determine upcoming, current, and past events based on the current date and time."""
    now = datetime.now()  # Get the current date and time

    upcoming_events = []
    current_events = []
    past_events = []

    for trainer in datafest.get("trainers", []):
        # Parse the event date
        event_date = datetime.strptime(trainer["date"], "%d %B")
        event_date = event_date.replace(year=now.year)  # Assume events are in the current year

        # Parse the event start and end times
        event_start_time = datetime.strptime(trainer["time"].split(" to ")[0], "%I %p").time()
        event_end_time = datetime.strptime(trainer["time"].split(" to ")[1], "%I %p").time()

        # Combine date and time for start and end
        event_start = datetime.combine(event_date, event_start_time)
        event_end = datetime.combine(event_date, event_end_time)

        # Categorize events based on the current date and time
        if event_start > now:
            upcoming_events.append(trainer)
        elif event_start <= now <= event_end:
            current_events.append(trainer)
        else:
            past_events.append(trainer)

    return {
        "upcoming": upcoming_events,
        "current": current_events,
        "past": past_events
    }

def handle_event_query(event_type):
    """Handle user queries for upcoming, current, or past events."""
    events = get_event_status()
    if event_type == "upcoming":
        return events["upcoming"]
    elif event_type == "current":
        return events["current"]
    elif event_type == "past":
        return events["past"]
    return []

def find_event_by_topic_or_course(query):
    """Find events by topic or course name."""
    events = get_event_status()
    all_events = events["upcoming"] + events["current"] + events["past"]

    for event in all_events:
        if query.lower() in event["topic"].lower():
            return event
    return None

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
            
            # Handle event-related queries
            if "upcoming events" in user_message.lower():
                upcoming_events = handle_event_query("upcoming")
                if upcoming_events:
                    response = "Our upcoming events are:\n" + "\n".join(
                        [f"'{event['topic']}' on {event['date']} by {event['name']}" for event in upcoming_events]
                    )
                else:
                    response = "There are no upcoming events."
                return JsonResponse({'text': response})

            elif "current events" in user_message.lower():
                current_events = handle_event_query("current")
                if current_events:
                    response = "Here are the current events:\n" + "\n".join(
                        [f"{event['name']} on {event['date']} at {event['time']} - {event['topic']}" for event in current_events]
                    )
                else:
                    response = "There are no ongoing events right now."
                return JsonResponse({'text': response})

            elif "past events" in user_message.lower():
                past_events = handle_event_query("past")
                if past_events:
                    response = "Here are the past events:\n" + "\n".join(
                        [f"{event['name']} on {event['date']} at {event['time']} - {event['topic']}" for event in past_events]
                    )
                else:
                    response = "There are no past events."
                return JsonResponse({'text': response})

            # Handle queries by topic or course
            event = find_event_by_topic_or_course(user_message)
            if event:
                if event in get_event_status()["upcoming"]:
                    response = f"Yes, '{event['topic']}' is an upcoming event. You can join it through our portal or YouTube channel."
                elif event in get_event_status()["current"]:
                    response = f"Yes, '{event['topic']}' is currently ongoing. You can join it through our portal or YouTube channel."
                else:
                    response = f"'{event['topic']}' was a past event. Stay tuned for similar events in the future!"
                return JsonResponse({'text': response})

            response = generate_nlp_response(user_message)
            #conversation_history.append((user_message, response)) # Store conversation history
            return JsonResponse({'text': response})
    return JsonResponse({'text': 'Invalid request'}, status=400)

def speak(text):
    """Convert chatbot's text response to speech with barge-in capability."""
    global is_speaking, current_speech
    
    def on_start(name):
        global is_speaking
        is_speaking = True
        
    def on_end(name, completed):
        global is_speaking, current_speech
        is_speaking = False
        current_speech = None
        time.sleep(1)
    
    # Stop any ongoing speech
    if is_speaking and current_speech:
        engine.stop()
    
    current_speech = text
    engine.connect('started-utterance', on_start)
    engine.connect('finished-utterance', on_end)
    engine.say(text)
    engine.runAndWait()

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

def listen():
    """Toggle microphone to listen continuously until user says 'bye' or 'exit'."""
    recognizer = sr.Recognizer() 
    mic_active = False 

    print("Press Enter to toggle the microphone on/off. Say 'bye' or 'exit' to stop completely.")
    
    while True:
        command = input("Press Enter to toggle mic or type 'exit' to quit: ").strip().lower()
        
        if command == "exit" or command == "bye" or command == "bye bye":
            conversation_history.clear() # Clear history
            print("Exiting the chat. Goodbye!")
        if command == "exit":
            print("Exiting the chat. Goodbye!")
            break 
        
        mic_active = not mic_active 
        
        if mic_active:
            print("Microphone is ON. Listening...")
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                
                while mic_active:
                    try:
                        print("Speak now...")
                        audio = recognizer.listen(source, phrase_time_limit=15) 
                        user_message = recognizer.recognize_google(audio) 
                        user_message = preprocess_recognized_text(user_message)
                        print(f"You said: {user_message}")
                        if "bye" in user_message.lower() or "exit" in user_message.lower() or "bye bye" in user_message.lower():
                            conversation_history.clear() # Clear history
                            print("Exiting the chat. Goodbye!")
                            mic_active = False  
                            break
                        request = HttpRequest()
                        request.method = 'POST'
                        request.body = json.dumps({'prompt': user_message}).encode('utf-8')
                    
                        response = get_response(request)
                        
                        # Extract the text from the JsonResponse
                        response_text = json.loads(response.content.decode('utf-8'))['text']
                        print(f"Chatbot: {response_text}")
                        speak(response_text)  # Speak the response aloud
                        
                    except sr.UnknownValueError:
                        print("Sorry, I did not understand that. Please try again.")
                    except sr.RequestError:
                        print("Sorry, there seems to be an issue with the speech recognition service.")
                    except sr.WaitTimeoutError:
                        print("Listening timed out. Please try again.")
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
