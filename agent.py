import os
from crewai import Agent , Task , Crew , LLM
from crewai_tools import YoutubeVideoSearchTool
from dotenv import load_dotenv


os.environ["OPENROUTER_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "sk-dummy"


llm = LLM(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    extra_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "CrewAI-Shorts-Agent"
    }
)

yt_tools = YoutubeVideoSearchTool()

researcher = Agent(
    role = "Youtube Trend Researcher",
    goal = "Find viral angles and key facts about {topic} for a YouTube Short",
    backstory = ("You are an expert at analyzing YouTube content. "
        "You know exactly what makes a video go viral. "
        "You dig deep into videos to find surprising facts that grab attention."),
    tools = [yt_tools],
    llm = llm,
    verbose = True,
    allow_delegations=False
)

research_task = Task(
    description=("1. Search YouTube for videos related to: {topic}. "
        "2. Extract 3 shocking facts or 'hidden gems' from these videos. "
        "3. Identify the common 'hook' used in popular videos on this topic."),
    expected_output="A bullet-point list of 3 facts and 1 recommended hook strategy.",
    agent=researcher
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task]
)

topic = "Games"

results = crew.kickoff(inputs={'topic':topic})

print(results)