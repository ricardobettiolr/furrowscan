import os

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Podés extender esta clase con más configuraciones si las necesitás

config = Config()
