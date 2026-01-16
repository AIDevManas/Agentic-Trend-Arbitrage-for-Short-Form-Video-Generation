import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import YoutubeVideoSearchTool
from dotenv import load_dotenv

load_dotenv()

# Setup Keys
if os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

def run_crew(topic: str):
    """
    Runs the crew and returns the raw outputs safely extracting them 
    from the CrewOutput object to avoid 'NoneType' errors.
    """
    
    # 1. SETUP
    llm = LLM(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    # Tool
    yt_tool = YoutubeVideoSearchTool()

    # 2. AGENTS
    researcher = Agent(role="Researcher", goal="Find viral facts", backstory="Youtube expert", tools=[yt_tool], llm=llm, verbose=True)
    scriptwriter = Agent(role="Scriptwriter", goal="Write 60s script", backstory="Copywriter", llm=llm, verbose=True)
    strategist = Agent(role="Strategist", goal="Metadata", backstory="SEO expert", llm=llm, verbose=True)

    # 3. TASKS
    task_research = Task(description=f"Find 3 shocking facts about {topic}.", expected_output="Bullet points.", agent=researcher)
    task_script = Task(description="Write a 60s script.", expected_output="Full script.", agent=scriptwriter, context=[task_research])
    task_strategy = Task(description="Generate Titles & Hashtags.", expected_output="Metadata.", agent=strategist, context=[task_script])

    # 4. RUN
    crew = Crew(
        agents=[researcher, scriptwriter, strategist],
        tasks=[task_research, task_script, task_strategy],
        verbose=True
    )
    
    # Capture the full CrewOutput object
    crew_output = crew.kickoff(inputs={'topic': topic})

    # 5. SAFE EXTRACTION
    # We inspect what kickoff returned to find the individual task outputs
    
    # Default values in case something fails
    research_out = "No output"
    script_out = "No output"
    strategy_out = "No output"

    # Try to extract from tasks_output list (Standard in new CrewAI versions)
    if hasattr(crew_output, 'tasks_output') and len(crew_output.tasks_output) >= 3:
        research_out = crew_output.tasks_output[0].raw
        script_out = crew_output.tasks_output[1].raw
        strategy_out = crew_output.tasks_output[2].raw
    
    # Fallback: If for some reason tasks_output is empty, try accessing the task objects directly
    # (Only works if the memory updated correctly)
    elif task_research.output and task_script.output:
        research_out = task_research.output.raw
        script_out = task_script.output.raw
        strategy_out = task_strategy.output.raw
    
    # Last resort: Just return the final result for all columns
    else:
        strategy_out = str(crew_output)

    return {
        "Research": research_out,
        "Script": script_out,
        "Strategy": strategy_out
    }