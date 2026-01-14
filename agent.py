import os
from crewai import Agent , Task , Crew , LLM
from crewai_tools import YoutubeVideoSearchTool
from dotenv import load_dotenv


load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")


os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


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

scriptwriter = Agent(
    role='Shorts Scriptwriter',
    goal='Write a punchy, under-60-second script',
    backstory="You are a viral copywriter. You write fast-paced scripts with strong hooks.",
    llm=llm,
    verbose=True
)


script_task = Task(
    description="Write a 60-second YouTube Short script using the research. Include visual cues [Visual: ...].",
    expected_output="A full script with Hook, Body, and Outro.",
    agent=scriptwriter,
    context=[research_task]
)

crew = Crew(
        agents=[researcher,scriptwriter],
        tasks=[research_task,script_task]
    )

topic = "India"

results = crew.kickoff(inputs={'topic':topic})

print(results)