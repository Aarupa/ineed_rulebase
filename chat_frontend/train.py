from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot(
    "MyBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri="sqlite:///db.sqlite3",  # Fresh DB
    read_only=False,  # Allow training (but we control it)
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "maximum_similarity_threshold": 0.90,
            
        }
    ]
)

trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("./greetings.yml", "./conversations.yml")  # Only your files