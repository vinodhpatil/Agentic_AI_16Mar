
# creating a resarch assistant
# installation core packages

!pip install -q crewai crewai-tools litellm

print(" core package installation complete")

# this demo is a single agent example using crew and groq

from crewai import Agent, Task, Crew, LLM
import os
import time
from getpass import getpass

# Get Groq API key
print("Enter your groq API key")
print("if you don't have one, get one at https://console.groq.com/keys")
groq_api_key = getpass("Groq API Key: ")

# set environment variable

os.environ["GROQ_API_KEY"] = groq_api_key

# initialize Groq LLM

llm = LLM(
    model="groq/llama-3.1-8b-instant", # fast model
    temperature=0.9,
    max_tokens=1000,
    api_key=groq_api_key
    #is_litellm=True
)

print("LLM initialized")

# create an agent

resarcher = Agent(
    role="Resarch Analyst",
    llm=llm,
    goal="Resarch some topic and provide clear, concise output",
    backstory="You are skilled resarcher which will provide accurate answer"
    #verbose=True
)

# create simple task

resarch_task = Task(
description="""Resarch the topic: {topic}

Provide brief overview in 4-5 sentences.
List the most important points.
""",

agent=resarcher,
expected_output="A concise 4-5 overview of the topic"

)

# create cew

crew = Crew(
    agents=[resarcher],
    tasks=[resarch_task]
    #verbose=True
)

# execute the task
print("\n starting resarch.....")
print("=" * 50)

result = crew.kickoff(
    inputs={
        "topic": "What is crew AI framework?" 
    }
)

print("=" * 50)
print("\n resarch complete")
print(result)
    
