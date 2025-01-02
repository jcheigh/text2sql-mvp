import os
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from abc import ABC, abstractmethod

import sys 
sys.path.append('../')
from helper import get_paths

class Config:
    """SQL agent config"""
    def __init__(
        self,
        openai_model = 'gpt-4-1106-preview',
        temperature  = 0.7
        ):
        path_map  = get_paths()
        env_fpath = path_map['env']
        sql_fpath = path_map['sql']

        load_dotenv(env_fpath)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.llm = ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            api_key=openai_api_key
            )
        self.db     = SQLDatabase.from_uri(f"sqlite:///{sql_fpath}")
        self.engine = create_engine(f"sqlite:///{sql_fpath}")

class Agent(ABC):
    """
    An abstract agent that answers queries (strings) by returning a float.
    """
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def run(self, user_query: str) -> float:
        """
        Takes a user query (natural language or SQL) and returns a numeric answer.
        This method must be overridden by concrete subclasses.
        """
        raise NotImplementedError

if __name__ == '__main__':
    pass 