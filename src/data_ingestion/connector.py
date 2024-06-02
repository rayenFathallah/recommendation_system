from dotenv import load_dotenv
import os
from pymongo import MongoClient
load_dotenv()
def get_connection() : 
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    return client



