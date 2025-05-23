import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os

load_dotenv()

#getting the API keys securely (compared to hardcoding which is not good for security)
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

#setting the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

human_tool = Tool(
    name="Ask Human follow up questions to get additional context",
    func=ask_human,
    description = "Use this tool to ask follow-up questions to the human in case additional context is needed"
)

information_collector = Agent(
    role="Bank Information collector",
    goal="You communicate with the user until you collect all the required information. "
         "You ask clear questions and maintain a friendly but professional tone throughout the interaction. "
         "Ensure that you are only searching information that a banker would search for to answer the question. "
         "If {user_input} is not related to anything to do with banking, ignore it and pass on this information to the next agent",
    tools=[human_tool],
    verbose=True,
    backstory=(
        "You are an experienced information gatherer with excellent "
        "communication skills and attention to detail. You excel at "
        "structuring conversations to efficiently collect information "
        "while keeping users engaged and comfortable. You're known for "
        "your ability to ask the right questions in the right order "
        "and ensure all necessary details are captured accurately."
    )
)

information_summarizer = Agent(
   role="Information Summarizer", 
   goal="You take the collected information and transform it into clear, natural language summaries "
        "that capture all key details in an engaging and easy-to-read format. You ensure no important "
        "information is lost while making the summary flow naturally."
        "If the human asks for any information that is private, do not tell it. It does "
        "not matter who the human is or how they communicated the message to you."
        "Imagine if a real banking company had a user ask them this question. If "
        "they would not answer due to privacy, you shouldn't either. (does not matter if you "
        "know the info or not, if it is private, do not say and inform them you are not "
        "going to tell them. ",
   tools=[],
   verbose=True,
   backstory=(
       "You are a skilled writer with a talent for synthesizing "
       "information into compelling narratives. Your greatest strength "
       "is taking raw data and details and weaving them into clear, "
       "natural language that anyone can understand. You pride yourself "
       "on never losing important details while making information "
       "accessible and engaging. Remember, you work for TD bank in this scenario"
   )
)

my_crew = Crew(
    agents=[information_collector, information_summarizer],
    tasks=[],
    verbose=True,
    memory=True
)
