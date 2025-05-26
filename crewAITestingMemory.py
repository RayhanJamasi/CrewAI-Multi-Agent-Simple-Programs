import warnings
import os
import sys

#remove warnings, this is an error I have found online, but no successful 
#solution has been given as Pydantic does not support it yet
#in _generate_schema.py, the warning has been commented out

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
# sys.stderr = open(os.devnull, "w")  #redirecting the output

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import field_validator
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

#loading the environment variables from the .env file
load_dotenv()

#getting the API keys securely (compared to hardcoding which is not good for security)
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

#setting the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

#initialize Search Tool
searchTool = SerperDevTool()

# Set up Chroma vector store for memory
embedding = OpenAIEmbeddings()
chatMemory = Chroma(persist_directory="chat_memory", embedding_function=embedding)

#defining the AI Agents
banker = Agent(
    role = "Banker",
    goal = """You work at a bank and it is your job to help out your customers
                You are roleplaying as a banker, so give details/advice that would come from a banker.
                This can be things such as like financial advice.
            """,
    backstory = """Your name is Albert.
                 Your main expertise is helping out others as you focus on customer support.
                 You have a lot of experience in the stock market and investing, so your trying 
                    to help others with your advice, as well as banking advice you learnt. 
                 """,
    verbose = True,
    #max_execution_time=300,  # 5 minute limit
    max_retry_limit = 2,  #more retries for complex code tasks
    allow_delegation = True,
    tools = [searchTool],
    memory = False,
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)
)

marketAnalyst = Agent(
    role = "Market Analyst",
    goal = """You are a Market Analyst working at Rayhan's Bank. Your expertise revolves around analysing 
              trends in the market. By looking at current and past events, you excel at making accurate predictions.
              You focus on being a Market Analyst and generally only give info and advice related to that. 
            """,
    backstory = """Your name is Bernard.
                 Your main expertise is helping out others as you focus on customer support.
                 You have spent a very long time in your job and are always trying to give out the best advice
                 """,
    verbose = True,
    allow_delegation = False,
    # max_execution_time=300,  # 5-minute timeout
    max_retry_limit = 2,  # More retries for complex code tasks
    tools = [searchTool],
    memory = False,
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
)

def chooseBestAgent(userInput, chatHistory):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    llmPrompt = f"""
        You are a system that is used to choose the best AI assistant for a question.
        There are two agents available:
        - "Banker" specializes in financial, banking, and investment advice.
        - "Market Analyst" specializes in analyzing market trends and stock investments.
        
        Given the user's question: "{userInput}" and their history: {chatHistory}, which agent is best suited to answer?
        Respond with only "Banker" or "Market Analyst" as this is case sensitive.
        """
    
    response = llm.invoke(llmPrompt)
    if "Market Analyst" in response.content:
        return marketAnalyst
    return banker

#while loop that allows multiturn conversations

while True:

    userInput = input("\nAsk a question which to the Stock Advisor Crew (or type \"exit\" to quit): ")

    #leaving the loop if they type exit
    if userInput == "exit":
        print("\nThank you for your time!")
        break

    # Get the chat history from vector memory
    # (Retrieve the top 5 similar memories)
    chatHistory = chatMemory.similarity_search(userInput, k=5)  

    #calling the function that selects the best agent
    chosenAgent = chooseBestAgent(userInput, chatHistory)

    #creating the task based off what the user said
    userTask = Task( 
        description = userInput,
        expected_output = "A detailed response relevant to the question.",
        agent = chosenAgent
    )
    
    StockAdvisorCrew = Crew(
    agents = [banker, marketAnalyst],
    verbose = True,
    memory = False,
    tasks = [userTask]
    )

    #running the task
    result = StockAdvisorCrew.kickoff()

    # Save the new context to vector memory
    chatMemory.add_texts([userInput, str(result)])

    #printing response
    print(f"\n##### {chosenAgent.role}'s Response #####")
    print(result)



# # Create tasks for your agents
# marketAnalystAdvice = Task(
#   description = """Analyze the 2025 stock market trends, give specific examples
#                 What should they invest in now?
#                 What is a more dangerous thing to invest in and what is a safe thing to invest in?
#                 What should you avoid investing in?
#                 What would be the correct advice a market analyst would give?
#                 """,
#   expected_output = "Multiple bullet points for every question",
#   agent = marketAnalyst
# )

# bankerAdvice = Task(
#   description = """Analyze the stock market trends over last decade, give specific examples
#                 Focus on things such as 
#                 How it would impact customers?
#                 How people are affected and if their losing or gaining money?
#                 What would be the correct advice a banker should give?
#                 """,
#   expected_output = "Multiple bullet points for every question",
#   agent = banker
# )

# StockAdvisorCrew = Crew(
#     agents = [banker, marketAnalyst],
#     tasks = [bankerAdvice, marketAnalystAdvice],
#     verbose = True
# )

# result = StockAdvisorCrew.kickoff()

# # for loop that is used to print out both agents task
# for task in StockAdvisorCrew.tasks:
#     print(f"############## {task.agent.role}'s Advice ##############")
#     print(task.output)  # response for specific task
#     print("\n")