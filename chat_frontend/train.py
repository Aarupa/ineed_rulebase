from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import os

script_dir = os.path.dirname(__file__)  # Directory of the current script
chatbot = ChatBot("MyBot", storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri="sqlite:///db.sqlite3")
trainer = ChatterBotCorpusTrainer(chatbot)

#Train chatbot with explicit corpus file paths
trainer.train(
    os.path.join(script_dir, "greetings.yml"),
    os.path.join(script_dir, "conversations.yml")
)

# trainer = ChatterBotCorpusTrainer(chatbot)
# trainer.train("chat_frontend\conversations.yml", 
#               "chat_frontend\greetings.yml")