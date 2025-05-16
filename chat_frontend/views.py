# # -------------------- Imports --------------------
# # Standard library imports
# import os
# import json
# import random
# import re
# import tempfile
# import logging
# from datetime import datetime, timedelta
# import string
# import threading
# import time

# # Third-party imports
# from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from fuzzywuzzy import process
# from pydub import AudioSegment
# from pydub.playback import play
# from gtts import gTTS
# import whisper
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin

# # Django imports
# from django.http import JsonResponse, HttpRequest
# from django.views.decorators.csrf import csrf_exempt
# from django.shortcuts import render

# import spacy
# import nltk
# import google.generativeai as genai

# # -------------------- Constants and Configurations --------------------
# # File paths
# script_dir = os.path.dirname(os.path.abspath(__file__))
# json_path = os.path.join(script_dir, 'json_files', 'content.json')
# dialogue_history_path = os.path.join(script_dir, 'json_files', 'history.json')

# # Initialize NLP and sentiment analysis tools
# nlp = spacy.load("en_core_web_sm")
# nltk.download('wordnet')
# sentiment_analyzer = SentimentIntensityAnalyzer()

# # Load FAQ data
# with open(json_path, 'r') as json_data:
#     faq_data = json.load(json_data)

# # Conversation history
# conversation_history = []
# history = {}
# if os.path.exists(dialogue_history_path):
#     with open(dialogue_history_path, 'r', encoding='utf-8') as f:
#         try:
#             history = json.load(f)
#         except json.JSONDecodeError:
#             history = {}

# # Clear history.json for one-time use
# if os.path.exists(dialogue_history_path):
#     with open(dialogue_history_path, 'w', encoding='utf-8') as f:
#         json.dump({}, f, indent=4, ensure_ascii=False)

# # Load Whisper model
# whisper_model = whisper.load_model("tiny")

# # Add chatbot's name
# CHATBOT_NAME = "Infi"

# # Load additional JSON files
# greetings_path = os.path.join(script_dir, 'json_files', 'greetings.json')
# farewells_path = os.path.join(script_dir, 'json_files', 'farewells.json')
# general_path = os.path.join(script_dir, 'json_files', 'general.json')

# with open(greetings_path, 'r', encoding='utf-8') as f:
#     greetings_data = json.load(f)

# with open(farewells_path, 'r', encoding='utf-8') as f:
#     farewells_data = json.load(f)

# with open(general_path, 'r', encoding='utf-8') as f:
#     general_data = json.load(f)

# # Gemini API configuration
# API_KEY = "AIzaSyA4bFTPKOQ3O4iKLmvQgys_ZjH_J1MnTUs"
# genai.configure(api_key=API_KEY)

# # -------------------- Website Crawler --------------------
# def crawl_website(base_url, max_pages=10):
#     """Crawl the website and index all page content"""
#     indexed_content = {}
#     visited = set()
#     to_visit = [base_url]
    
#     while to_visit and len(visited) < max_pages:
#         url = to_visit.pop()
#         try:
#             response = requests.get(url, timeout=10)
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Store page content with URL as key
#             indexed_content[url] = {
#                 'title': soup.title.string if soup.title else 'No Title',
#                 'text': ' '.join(soup.stripped_strings),
#                 'links': []
#             }
            
#             # Extract and queue new links
#             for link in soup.find_all('a', href=True):
#                 absolute_url = urljoin(base_url, link['href'])
#                 if absolute_url.startswith(base_url) and absolute_url not in visited:
#                     indexed_content[url]['links'].append(absolute_url)
#                     if absolute_url not in to_visit:
#                         to_visit.append(absolute_url)
            
#             visited.add(url)
#         except Exception as e:
#             print(f"Error crawling {url}: {str(e)}")
    
#     return indexed_content

# # Initialize website index
# WEBSITE_INDEX = crawl_website("https://indeedinspiring.com")

# # Background website content refresher
# def refresh_website_content():
#     """Periodically update website content"""
#     while True:
#         global WEBSITE_INDEX
#         WEBSITE_INDEX = crawl_website("https://indeedinspiring.com")
#         time.sleep(86400)  # Refresh daily

# refresh_thread = threading.Thread(target=refresh_website_content, daemon=True)
# refresh_thread.start()

# # -------------------- Utility Functions --------------------
# def save_conversation_to_file(user_message, response):
#     """Save the conversation to a JSON file as key-value pairs."""
#     history[user_message] = response
#     with open(dialogue_history_path, 'w', encoding='utf-8') as f:
#         json.dump(history, f, indent=4, ensure_ascii=False)

# def correct_spelling(query):
#     """Correct spelling mistakes in the user query using TextBlob."""
#     blob = TextBlob(query)
#     return str(blob.correct())

# def get_best_match(query, choices, threshold=80, min_length=2):
#     """Find the best matching FAQ keyword using fuzzy string matching."""
#     if len(query) < min_length:
#         return None
#     best_match, score = process.extractOne(query, choices)
#     return best_match if score >= threshold else None

# def analyze_sentiment(msg):
#     """Analyze the sentiment of a user message using VaderSentiment."""
#     score = sentiment_analyzer.polarity_scores(msg)['compound']
#     if score >= 0.5:
#         return "positive"
#     elif score <= -0.5:
#         return "negative"
#     return "neutral"

# def enhance_speech_recognition(text):
#     """Advanced correction for Indeed Inspiring Infotech variations"""
#     # Phonetic correction mapping with priority
#     corrections = [
#         (r"\b(indian|india|indexing|indeedin|indie|indigo|indenting|indentation)\s*(inspiring|spring|spire|spirit|sparring)\s*(infotech|info tech|in tech|inft|tech)?\b", 
#          "Indeed Inspiring Infotech"),
#         (r"\b(indeed|ended|indoor|indie)\s*(inspiring|spring|spire|spirit)\b", 
#          "Indeed Inspiring"),
#         (r"\b(indeed|indian)\s*(infotech|info tech)\b", 
#          "Indeed Infotech"),
#         (r"\b(inspiring|spring)\s*(infotech)\b", 
#          "Inspiring Infotech"),
#         (r"\b(indeed inspiring|indian inspiring)\b", 
#          "Indeed Inspiring Infotech")
#     ]
    
#     # Apply corrections in priority order
#     for pattern, replacement in corrections:
#         text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
#     # Final standardization
#     text = re.sub(r"Indeed(\s*Inspiring)?(\s*Infotech)?", 
#                  "Indeed Inspiring Infotech", text, flags=re.IGNORECASE)
    
#     return text.strip()

# # Update preprocess_recognized_text to use the enhanced version
# preprocess_recognized_text = enhance_speech_recognition

# # -------------------- Chatbot Logic --------------------
# def classify_query(msg):
#     """Classify the query as company-related (FAQ) or general conversation."""
#     # Normalize the message by removing punctuation and converting to lowercase
#     msg_normalized = msg.translate(str.maketrans('', '', string.punctuation)).lower()

#     # Handle "What is your name?" query
#     if "what is your name" in msg_normalized or "your name" in msg_normalized:
#         return "general_convo", f"My name is {CHATBOT_NAME}."

#     greetings = [f"hey {CHATBOT_NAME.lower()}", f"hi {CHATBOT_NAME.lower()}","hi ","Hi ","hi  ","Hi  ","hello ","Hello ","hey ","Hey ","hii ","Hii ","hii  ","Hii  "]

#     if any(greeting in msg_normalized for greeting in greetings):
#         return "greeting", f"Hello! I'm {CHATBOT_NAME}. How can I assist you today?"

#     for faq in faq_data['faqs']:
#         if msg_normalized == faq['question'].translate(str.maketrans('', '', string.punctuation)).lower():
#             return "company", random.choice(faq['responses'])

#     keywords = [keyword.translate(str.maketrans('', '', string.punctuation)).lower() for faq in faq_data['faqs'] for keyword in faq['keywords']]
#     best_match = get_best_match(msg_normalized, keywords)

#     corrected_msg = correct_spelling(msg_normalized)
#     if corrected_msg != msg_normalized:
#         for faq in faq_data['faqs']:
#             if best_match in faq['keywords']:
#                 suggested_question = faq['question']
#                 response = random.choice(faq['responses'])
#                 return "company", f"Did you mean '{suggested_question}'?\n {response}"

#     if best_match:
#         for faq in faq_data['faqs']:
#             if best_match in faq['keywords']:
#                 return "company", random.choice(faq['responses'])

#     # If no match is found, return a default response
#     return None, None

# def generate_nlp_response(msg):
#     """Generate a basic NLP response for general conversation."""
#     doc = nlp(msg)
#     if any(token.lower_ in ["hi ", "hello", "hey", "hii"] for token in doc):
#         return random.choice(["Hey there! How's your day going?", "Hello! What's up?", "Hi! How can I assist you today?"])
#     elif "how are you" in msg.lower():
#         return random.choice(["I'm doing great, thanks for asking! How about you?", "I'm good! Hope you're having a great day too."])
#     elif msg.lower() in ["great", "good", "awesome", "fantastic", "amazing"]:
#         return random.choice(["Glad to hear that! What's on your mind?", "That's awesome! How can I assist you today?"])
#     elif "thank you" in msg.lower() or "thanks" in msg.lower():
#         return random.choice(["You're very welcome!", "Anytime! Glad I could help."])
#     elif msg.lower() in ["bye", "exit"]:
#         conversation_history.clear()
#         return "Ok bye! Have a good day!"
#     else:
#         return None

# def get_contextual_response(user_message):
#     """Generate a contextual response based on the user's message."""
#     if "weather" in user_message.lower():
#         return "I'm not equipped to provide weather updates, but you can check a weather app!"
#     elif "time" in user_message.lower():
#         return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
#     return None

# def handle_time_based_greeting(msg):
#     """Handle time-based greetings and provide an appropriate response."""
#     greetings = ["good morning", "good afternoon", "good evening", "good night"]
#     msg_lower = msg.lower()

#     # Check if the message contains a time-based greeting
#     for greeting in greetings:
#         if greeting in msg_lower:
#             current_hour = datetime.now().hour
#             if greeting == "good morning":
#                 if current_hour < 12:
#                     return "Good morning! How can I assist you today?"
#                 elif current_hour < 18:
#                     return "It's already afternoon, but good day to you!"
#                 else:
#                     return "It's evening now, but good day to you!"
#             elif greeting == "good afternoon":
#                 if current_hour < 12:
#                     return "It's still morning, but good day to you!"
#                 elif current_hour < 18:
#                     return "Good afternoon! How can I assist you today?"
#                 else:
#                     return "It's evening now, but good day to you!"
#             elif greeting == "good evening":
#                 if current_hour < 12:
#                     return "It's still morning, but good day to you!"
#                 elif current_hour < 18:
#                     return "It's still afternoon, but good day to you!"
#                 else:
#                     return "Good evening! How can I assist you today?"
#             elif greeting == "good night":
#                 return "Good night! Sleep well and take care!"

#     # Handle queries about the current time
#     if "current time" in msg_lower or "current time" in msg_lower:
#         return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

#     # Fallback to classify query or ChatterBot response
#     category, response = classify_query(msg)
#     if response:
#         return response
    
# def handle_date_related_queries(msg):
#     """Handle date-related queries and provide an appropriate response."""
#     msg_lower = msg.lower()
#     today = datetime.now()
    
#     # Define a mapping for generic conditions
#     date_mapping = {
#         "today": today,
#         "tomorrow": today + timedelta(days=1),
#         "day after tomorrow": today + timedelta(days=2),
#         "yesterday": today - timedelta(days=1),
#         "day before yesterday": today - timedelta(days=2),
#         "next week": today + timedelta(weeks=1),
#         "last week": today - timedelta(weeks=1),
#         "next month": (today.replace(day=28) + timedelta(days=4)).replace(day=1),
#         "last month": (today.replace(day=1) - timedelta(days=1)).replace(day=1),
#         "next year": today.replace(year=today.year + 1),
#         "last year": today.replace(year=today.year - 1)
#     }
    
#     # Check for specific phrases in the message
#     for key, date in date_mapping.items():
#         if key in msg_lower:
#             if "date" in msg_lower:
#                 return f"The {key}'s date is {date.strftime('%B %d, %Y')}."
#             elif "day" in msg_lower:
#                 return f"The {key} is {date.strftime('%A')}."
    
#     # Fallback for unrecognized queries
#     return None

# def get_priority_response(preprocessed_input):
#     """Check if the input matches greetings, farewells, or general responses."""
#     # Normalize input by removing punctuation and converting to lowercase
#     normalized_input = preprocessed_input.translate(str.maketrans('', '', string.punctuation)).lower()
#     for category, data in [("greetings", greetings_data["greetings"]),
#                            ("farewells", farewells_data["farewells"]),
#                            ("general", general_data["general"])]:
#         if normalized_input in map(str.lower, data["inputs"]):
#             logging.debug(f"Matched {category} for input: {normalized_input}")
#             return random.choice(data["responses"])
#     logging.debug(f"No match found in priority responses for input: {normalized_input}")
#     return None

# # -------------------- Enhanced Gemini Integration --------------------
# def get_gemini_response_with_context(user_query):
#     """Get response using Gemini with website context"""
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
        
#         # Find most relevant page content
#         best_match = None
#         best_score = 0
#         query_keywords = set(user_query.lower().split())
        
#         for url, data in WEBSITE_INDEX.items():
#             page_keywords = set(data['text'].lower().split())
#             match_score = len(query_keywords & page_keywords)
#             if match_score > best_score:
#                 best_match = data
#                 best_score = match_score
        
#         # Prepare context
#         context = ""
#         if best_match and best_score > 2:  # Minimum keyword matches
#             context = f"Relevant page content from {best_match['title']}:\n{best_match['text'][:2000]}..."
        
#         prompt = f"""You are a support assistant for Indeed Inspiring Infotech.
#         Strict Rules:
#         1. ONLY use information from the provided context
#         2. If unsure, only say "I am still learning "
#         3. Keep responses concise (1-2 sentences)
#         4. Always include the source URL if available
        
#         Context: {context}
        
#         User Query: {user_query}
        
#         Response:"""
        
#         response = model.generate_content(prompt)
#         return response.text
        
#     except Exception as e:
#         return f"I encountered an error: {str(e)}"

# # -------------------- Unified Response Handler --------------------
# def get_final_response(user_input):
#     """Complete response pipeline with website fallback"""
#     # 1. Preprocess input
#     processed_input = preprocess_recognized_text(user_input.lower())
    
#     # 2. Try direct company responses first
#     company_response = get_company_response(processed_input)
#     if company_response:
#         return company_response
    
#     # 3. Handle special cases (time, greetings etc.)
#     special_handlers = [
#         handle_date_related_queries,
#         handle_time_based_greeting,
#         generate_nlp_response,
#         get_contextual_response,
#         lambda x: classify_query(x)[1],  # Get just the response
#         get_priority_response
#     ]
    
#     for handler in special_handlers:
#         response = handler(processed_input)
#         if response:
#             return response
    
#     # 4. Final fallback to Gemini with website context
#     return get_gemini_response_with_context(processed_input)

# def get_company_response(query):
#     """Handle company-specific questions with high accuracy"""
#     # Standardize the query first
#     query = enhance_speech_recognition(query.lower())
    
#     # Enhanced FAQ matching
#     faq_mapping = {
#         r"(founder|who\s*started|ceo|owner)": 
#             "Indeed Inspiring Infotech was founded by Mr. Kushal Sharma.",
#         r"(contact|reach|phone|email|address)": 
#             "You can contact us at:\nPhone: +1 (123) 456-7890\nEmail: info@indeedinspiring.com",
#         r"(services|offerings|what\s*you\s*do)":
#             "We specialize in AI solutions, web development, and digital transformation services.",
#         r"(about|information|tell\s*me\s*about)":
#             "Indeed Inspiring Infotech is a technology company focused on innovative IT solutions."
#     }
    
#     for pattern, response in faq_mapping.items():
#         if re.search(pattern, query):
#             return response
    
#     return None

# # -------------------- HTTP Request Handlers --------------------
# @csrf_exempt
# def get_response(request):
#     """Handle HTTP requests and return chatbot responses as JSON."""
#     global conversation_history
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         user_message = data.get('prompt', '')
#         logging.debug(f"User message: {user_message}")
#         if user_message:
#             # Get response from unified handler
#             bot_response = get_final_response(user_message)
            
#             # Save to history
#             conversation_history.append((user_message, bot_response))
#             save_conversation_to_file(user_message, bot_response)
            
#             return JsonResponse({'text': bot_response})
    
#     return JsonResponse({'text': 'Invalid request'}, status=400)

# @csrf_exempt
# def clear_history(request):
#     """Clear the conversation history."""
#     global conversation_history
#     if request.method == 'POST':
#         conversation_history.clear()
#         return JsonResponse({'status': 'success', 'message': 'Conversation history cleared.'})
#     return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

# def chat(request):
#     """Render the chatbot HTML template."""
#     try:
#         return render(request, 'chatbot.html')
#     except Exception as e:
#         logging.error(f"Error loading template: {e}")
#         return JsonResponse({'error': str(e)}, status=500)

# # -------------------- Speech and Listening Functions --------------------
# def speak(text):
#     """Convert chatbot's text response to speech using gTTS."""
#     try:
#         tts = gTTS(text=text, lang='en')
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
#             tts.save(temp_audio_file.name)
#             temp_audio_path = temp_audio_file.name
#         audio = AudioSegment.from_file(temp_audio_path, format="mp3")
#         play(audio)
#         os.remove(temp_audio_path)
#     except Exception as e:
#         print(f"An error occurred during text-to-speech conversion: {e}")

# def listen():
#     """Enhanced voice interaction with better error handling"""
#     import sounddevice as sd
#     from scipy.io.wavfile import write
    
#     print("Voice interaction ready. Say 'exit' to quit.")
    
#     while True:
#         try:
#             # Record audio
#             duration = 5  # Shorter chunks for faster response
#             print("\nListening... (speak now)")
#             audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
#             sd.wait()
            
#             # Save and transcribe
#             with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
#                 write(tmpfile.name, 16000, audio)
#                 raw_text = whisper_model.transcribe(tmpfile.name)["text"]
            
#             if not raw_text.strip():
#                 continue
                
#             # Enhanced processing
#             processed_text = preprocess_recognized_text(raw_text)
#             print(f"You said: {processed_text}")
            
#             # Check for exit command
#             if any(cmd in processed_text.lower() for cmd in ["exit", "bye", "stop"]):
#                 print("Goodbye!")
#                 break
                
#             # Get and speak response
#             response = get_final_response(processed_text)
#             print(f"Assistant: {response}")
#             speak(response)
            
#         except KeyboardInterrupt:
#             print("\nSession ended")
#             break
#         except Exception as e:
#             print(f"Error: {e}")
#             speak("Sorry, I encountered an error. Please try again.")



# # Add this near your other imports
# trees_path = os.path.join(script_dir, 'json_files', 'trees.json')

# # Load trees data if the file exists
# trees_data = {}
# if os.path.exists(trees_path):
#     with open(trees_path, 'r', encoding='utf-8') as f:
#         try:
#             trees_data = json.load(f)
#         except json.JSONDecodeError:
#             trees_data = {}

# # -------------------- Give Me Trees Foundation Handler --------------------
# @csrf_exempt
# def gmtt_response(request):
#     """Handle requests specifically for Give Me Trees Foundation"""
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             user_message = data.get('prompt', '').lower()
            
#             # Check if this is a trees-related query
#             tree_keywords = ["tree", "plant", "forest", "environment", "gmtt", "give me trees"]
#             if not any(keyword in user_message for keyword in tree_keywords):
#                 return JsonResponse({
#                     'text': "I'm the Give Me Trees specialist. Please ask me about tree planting or environmental conservation."
#                 })
            
#             # Try to get response from trees.json first
#             response = get_trees_response(user_message)
#             if response:
#                 return JsonResponse({'text': response})
            
#             # Fallback to Gemini with trees context
#             gemini_response = get_gemini_trees_response(user_message)
#             return JsonResponse({'text': gemini_response})
            
#         except Exception as e:
#             logging.error(f"Error in gmtt_response: {e}")
#             return JsonResponse({
#                 'text': "Sorry, I encountered an error processing your trees-related request."
#             }, status=500)
    
#     return JsonResponse({'text': 'Invalid request method'}, status=400)

# def get_trees_response(query):
#     """Get response from trees.json data"""
#     if not trees_data:
#         return None
        
#     # Standardize query
#     query = query.translate(str.maketrans('', '', string.punctuation)).lower()
    
#     # Check exact matches first
#     for item in trees_data.get('faqs', []):
#         if query == item['question'].lower():
#             return random.choice(item['responses'])
    
#     # Check keywords
#     best_match = None
#     best_score = 0
    
#     for item in trees_data.get('faqs', []):
#         for keyword in item.get('keywords', []):
#             score = fuzz.ratio(query, keyword.lower())
#             if score > best_score and score > 70:  # 70% match threshold
#                 best_score = score
#                 best_match = item
    
#     if best_match:
#         return random.choice(best_match['responses'])
    
#     return None

# def get_gemini_trees_response(query):
#     """Get trees-specific response from Gemini"""
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
        
#         # Basic trees context
#         context = """
#         Give Me Trees Foundation is a non-profit organization dedicated to environmental conservation through tree planting.
#         Key Information:
#         - Founded in 1978 by Kuldeep Bharti
#         - Headquarters in New Delhi, India
#         - Planted over 20 million trees
#         - Focus areas: Urban greening, rural afforestation, environmental education
#         - Website: https://www.givemetrees.org
#         """
        
#         prompt = f"""You are a specialist assistant for Give Me Trees Foundation.
#         Rules:
#         1. Only provide information about tree planting and environmental conservation
#         2. For other topics, respond "I specialize in tree-related questions"
#         3. Keep responses under 3 sentences
#         4. Always mention the website givemetrees.org
        
#         Context: {context}
        
#         Query: {query}
        
#         Response:"""
        
#         response = model.generate_content(prompt)
#         return response.text
        
#     except Exception as e:
#         return "I'm having trouble accessing tree information right now. Please visit givemetrees.org for details."
    
# def gmtt(request):
#     """Render the chatbot HTML template."""
#     try:
#         return render(request, 'give_me_tree.html')
#     except Exception as e:
#         logging.error(f"Error loading template: {e}")
#         return JsonResponse({'error': str(e)}, status=500)



# -------------------- Imports --------------------
# Standard library imports
import os
import json
import random
import re
import tempfile
import logging
from datetime import datetime, timedelta
import string
import threading
import time

# Third-party imports
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import fuzz, process
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
import whisper
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Django imports
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

import spacy
import nltk
import google.generativeai as genai

# -------------------- Constants and Configurations --------------------
# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'json_files', 'content.json')
dialogue_history_path = os.path.join(script_dir, 'json_files', 'history.json')
greetings_path = os.path.join(script_dir, 'json_files', 'greetings.json')
farewells_path = os.path.join(script_dir, 'json_files', 'farewells.json')
general_path = os.path.join(script_dir, 'json_files', 'general.json')
trees_path = os.path.join(script_dir, 'json_files', 'trees.json')

# Initialize NLP and sentiment analysis tools
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load data files
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

faq_data = load_json_file(json_path)
greetings_data = load_json_file(greetings_path)
farewells_data = load_json_file(farewells_path)
general_data = load_json_file(general_path)
trees_data = load_json_file(trees_path)

# Conversation history
conversation_history = []
history = load_json_file(dialogue_history_path)

# Clear history.json for one-time use
with open(dialogue_history_path, 'w', encoding='utf-8') as f:
    json.dump({}, f, indent=4, ensure_ascii=False)

# Load Whisper model
whisper_model = whisper.load_model("tiny")

# Chatbot names
CHATBOT_NAME = "Infi"
GMTT_NAME = "Infi"

# Gemini API configuration
API_KEY = "AIzaSyA4bFTPKOQ3O4iKLmvQgys_ZjH_J1MnTUs"
genai.configure(api_key=API_KEY)

# -------------------- Website Crawler --------------------
def crawl_website(base_url, max_pages=10):
    """Crawl the website and index all page content"""
    indexed_content = {}
    visited = set()
    to_visit = [base_url]
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            indexed_content[url] = {
                'title': soup.title.string if soup.title else 'No Title',
                'text': ' '.join(soup.stripped_strings),
                'links': []
            }
            
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(base_url, link['href'])
                if absolute_url.startswith(base_url) and absolute_url not in visited:
                    indexed_content[url]['links'].append(absolute_url)
                    if absolute_url not in to_visit:
                        to_visit.append(absolute_url)
            
            visited.add(url)
        except Exception as e:
            logging.error(f"Error crawling {url}: {str(e)}")
    
    return indexed_content

# Initialize website indexes
INDEED_INDEX = crawl_website("https://indeedinspiring.com")
GMTT_INDEX = crawl_website("https://www.givemetrees.org")

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
    """Find the best matching keyword using fuzzy string matching."""
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

def enhance_speech_recognition(text):
    """Advanced correction for speech recognition variations"""
    corrections = [
        (r"\b(indian|india|indeedin|indie)\s*(inspiring|spring)\s*(infotech|info tech)?\b", 
         "Indeed Inspiring Infotech"),
        (r"\b(indeed|indian)\s*(infotech|info tech)\b", 
         "Indeed Infotech"),
        (r"\b(give me|get me|i need)\s*(trees|tree|plants)\b",
         "Give Me Trees"),
        (r"\b(environment|eco|green)\s*(foundation|org|charity)\b",
         "Give Me Trees Foundation")
    ]
    
    for pattern, replacement in corrections:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()

# -------------------- Shared Conversation Handlers --------------------
def handle_time_based_greeting(msg):
    """Handle time-based greetings and provide an appropriate response."""
    greetings = ["good morning", "good afternoon", "good evening", "good night"]
    msg_lower = msg.lower()

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

    if "current time" in msg_lower:
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    return None

def handle_date_related_queries(msg):
    """Handle date-related queries and provide an appropriate response."""
    msg_lower = msg.lower()
    today = datetime.now()
    
    date_mapping = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "day after tomorrow": today + timedelta(days=2),
        "yesterday": today - timedelta(days=1),
        "day before yesterday": today - timedelta(days=2),
        "next week": today + timedelta(weeks=1),
        "last week": today - timedelta(weeks=1),
        "next month": (today.replace(day=28) + timedelta(days=4)).replace(day=1),
        "last month": (today.replace(day=1) - timedelta(days=1)).replace(day=1),
        "next year": today.replace(year=today.year + 1),
        "last year": today.replace(year=today.year - 1)
    }
    
    for key, date in date_mapping.items():
        if key in msg_lower:
            if "date" in msg_lower:
                return f"The {key}'s date is {date.strftime('%B %d, %Y')}."
            elif "day" in msg_lower:
                return f"The {key} is {date.strftime('%A')}."
    
    return None

def generate_nlp_response(msg):
    """Generate a basic NLP response for general conversation."""
    doc = nlp(msg)
    greetings = ["hi", "hello", "hey", "hii"]
    
    if any(token.lower_ in greetings for token in doc):
        return random.choice([f"Hello! I'm {CHATBOT_NAME}. How can I help you today?", 
                             f"Hi there! I'm {GMTT_NAME} here to help with tree conservation!"])
    
    # Rest of the function remains the same...
    elif "how are you" in msg.lower():
        return random.choice(["I'm doing great, thanks for asking!", "I'm good! How about you?"])
    elif msg.lower() in ["great", "good", "awesome"]:
        return random.choice(["Glad to hear that!", "That's wonderful!"])
    elif "thank you" in msg.lower() or "thanks" in msg.lower():
        return random.choice(["You're welcome!", "Happy to help!"])
    elif "bye" in msg.lower() or "exit" in msg.lower():
        conversation_history.clear()
        return "Goodbye! Have a great day!"
    
    return None

def get_priority_response(preprocessed_input):
    """Check if the input matches greetings, farewells, or general responses."""
    normalized_input = preprocessed_input.translate(str.maketrans('', '', string.punctuation)).lower()
    
    for category, data in [("greetings", greetings_data.get("greetings", {})),
                           ("farewells", farewells_data.get("farewells", {})),
                           ("general", general_data.get("general", {}))]:
        if normalized_input in map(str.lower, data.get("inputs", [])):
            return random.choice(data.get("responses", []))
    
    return None

def handle_general_conversation(user_message):
    """Handle general conversation that both chatbots should respond to"""
    handlers = [
        handle_time_based_greeting,
        handle_date_related_queries,
        generate_nlp_response,
        get_priority_response
    ]
    
    for handler in handlers:
        response = handler(user_message)
        if response:
            return response
    
    return None

# -------------------- Indeed Inspiring Chatbot --------------------
def get_company_response(query):
    """Handle company-specific questions"""
    query = enhance_speech_recognition(query.lower())
    
    faq_mapping = {
        r"(founder|who\s*started|ceo|owner)": 
            "Indeed Inspiring Infotech was founded by Mr. Kushal Sharma.",
        r"(contact|reach|phone|email|address)": 
            "You can contact us at:\nPhone: +1 (123) 456-7890\nEmail: info@indeedinspiring.com",
        r"(services|offerings|what\s*you\s*do)":
            "We specialize in AI solutions, web development, and digital transformation services.",
        r"(about|information|tell\s*me\s*about)":
            "Indeed Inspiring Infotech is a technology company focused on innovative IT solutions."
    }
    
    for pattern, response in faq_mapping.items():
        if re.search(pattern, query):
            return response
    
    return None

def get_gemini_indeed_response(user_query):
    """Get response using Gemini with Indeed website context"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        best_match = None
        best_score = 0
        query_keywords = set(user_query.lower().split())
        
        for url, data in INDEED_INDEX.items():
            page_keywords = set(data['text'].lower().split())
            match_score = len(query_keywords & page_keywords)
            if match_score > best_score:
                best_match = data
                best_score = match_score
        
        context = ""
        if best_match and best_score > 2:
            context = f"Relevant page content from {best_match['title']}:\n{best_match['text'][:2000]}..."
        
        prompt = f"""You are a support assistant for Indeed Inspiring Infotech.
        Rules:
        1. ONLY use information from the provided context
        2. If unsure, say "I'm still learning about Indeed Inspiring Infotech"
        3. Keep responses concise (1-2 sentences)
        4. Always include the source URL if available
        
        Context: {context}
        
        User Query: {user_query}
        
        Response:"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"I encountered an error: {str(e)}"

def get_indeed_response(user_input):
    """Complete Indeed chatbot response pipeline"""
    # Handle name question first
    if "what is your name" in user_input.lower() or "your name" in user_input.lower():
        return f"My name is {CHATBOT_NAME}. How can I assist you with Indeed Inspiring Infotech?"
    
    # 1. Handle general conversation
    general_response = handle_general_conversation(user_input)
    if general_response:
        return general_response
    
    # 2. Check for company-specific responses
    company_response = get_company_response(user_input)
    if company_response:
        return company_response
    
    # 3. Fallback to Gemini with Indeed context
    return get_gemini_indeed_response(user_input)

# -------------------- Give Me Trees Chatbot --------------------
def get_trees_response(query):
    """Get response from trees.json data"""
    if not trees_data:
        return None
        
    query = query.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Check exact matches
    for item in trees_data.get('faqs', []):
        if query == item['question'].lower():
            return random.choice(item['responses'])
    
    # Check keywords
    best_match = None
    best_score = 0
    
    for item in trees_data.get('faqs', []):
        for keyword in item.get('keywords', []):
            score = fuzz.ratio(query, keyword.lower())
            if score > best_score and score > 70:
                best_score = score
                best_match = item
    
    if best_match:
        return random.choice(best_match['responses'])
    
    return None

def get_gemini_gmtt_response(user_query):
    """Get trees-specific response from Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        best_match = None
        best_score = 0
        query_keywords = set(user_query.lower().split())
        
        for url, data in GMTT_INDEX.items():
            page_keywords = set(data['text'].lower().split())
            match_score = len(query_keywords & page_keywords)
            if match_score > best_score:
                best_match = data
                best_score = match_score
        
        context = """
        Give Me Trees Foundation is a non-profit organization dedicated to environmental conservation through tree planting.
        Key Information:
        - Founded in 1978 by Kuldeep Bharti
        - Headquarters in New Delhi, India
        - Planted over 20 million trees
        - Focus areas: Urban greening, rural afforestation, environmental education
        - Website: https://www.givemetrees.org
        """
        
        if best_match and best_score > 2:
            context += f"\n\nRelevant page content from {best_match['title']}:\n{best_match['text'][:2000]}..."
        
        prompt = f"""You are a specialist assistant for Give Me Trees Foundation.
        Rules:
        1. Only provide information about tree planting and environmental conservation
        2. For other topics, respond "I specialize in tree-related questions"
        3. Keep responses under 3 sentences
        4. Always mention the website givemetrees.org
        
        Context: {context}
        
        Query: {user_query}
        
        Response:"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return "I'm having trouble accessing tree information right now. Please visit givemetrees.org for details."

def get_gmtt_response(user_input):
    """Complete GMTT chatbot response pipeline"""
    # Handle name question first
    if "what is your name" in user_input.lower() or "your name" in user_input.lower():
        return f"My name is {GMTT_NAME}. I'm here to help with Give Me Trees Foundation queries."
    
    # 1. Handle general conversation
    general_response = handle_general_conversation(user_input)
    if general_response:
        return general_response
    
    # 2. Check if this is a trees-related query
    tree_keywords = ["tree", "plant", "forest", "environment", "gmtt", "give me trees"]
    if not any(keyword in user_input.lower() for keyword in tree_keywords):
        return f"I'm {GMTT_NAME}, the Give Me Trees specialist. How can I help you with tree planting or environmental conservation today?"
    
    # 3. Check for trees-specific responses
    trees_response = get_trees_response(user_input)
    if trees_response:
        return trees_response
    
    # 4. Fallback to Gemini with trees context
    return get_gemini_gmtt_response(user_input)

# -------------------- HTTP Request Handlers --------------------
@csrf_exempt
def get_response(request):
    """Indeed chatbot endpoint"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('prompt', '')
            
            if user_message:
                bot_response = get_indeed_response(user_message)
                save_conversation_to_file(user_message, bot_response)
                return JsonResponse({'text': bot_response})
                
        except Exception as e:
            logging.error(f"Error in Indeed response: {e}")
            return JsonResponse({'text': 'Sorry, I encountered an error'}, status=500)
    
    return JsonResponse({'text': 'Invalid request'}, status=400)

@csrf_exempt
def gmtt_response(request):
    """Give Me Trees chatbot endpoint"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('prompt', '')
            
            if user_message:
                bot_response = get_gmtt_response(user_message)
                return JsonResponse({'text': bot_response})
                
        except Exception as e:
            logging.error(f"Error in GMTT response: {e}")
            return JsonResponse({
                'text': "Sorry, I encountered an error processing your trees-related request."
            }, status=500)
    
    return JsonResponse({'text': 'Invalid request method'}, status=400)

def chat(request):
    """Render Indeed chatbot interface"""
    return render(request, 'chatbot.html')

def gmtt(request):
    """Render GMTT chatbot interface"""
    return render(request, 'give_me_tree.html')

# -------------------- Background Services --------------------
def refresh_website_content():
    """Periodically update website content for both chatbots"""
    while True:
        global INDEED_INDEX, GMTT_INDEX
        INDEED_INDEX = crawl_website("https://indeedinspiring.com")
        GMTT_INDEX = crawl_website("https://www.givemetrees.org")
        time.sleep(86400)  # Refresh daily

# Start background thread
refresh_thread = threading.Thread(target=refresh_website_content, daemon=True)
refresh_thread.start()