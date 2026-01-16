import chainlit as cl
from agent import run_crew
import pandas as pd

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="""🎬 **Welcome to AutoShorts AI!**\n\nI am your agent team. 
                     Give me a topic (e.g., *'The history of tea'*), and I will research, script, and package a YouTube Short for you.""").send()
    
@cl.on_message
async def main(message:cl.Message):
    topic = message.content

    msg = cl.Message(content=f"🕵️ **Researching:** *{topic}*... \n*(Please wait, agents are working)*")
    await msg.send()

    try:
        outputs = run_crew(topic)
        
        data = {
            "Step 1: Research": [outputs["Research"]],
            "Step 2: Script": [outputs["Script"]],
            "Step 3: Strategy": [outputs["Strategy"]]
        }

        df = pd.DataFrame(data)

        await cl.Message(
            content=f"✅ **Job Complete!**\nSee the 3-column breakdown below:",
            elements=[
                cl.Dataframe(
                    name="Agent Workflow Table",
                    data=df,
                    display="inline"
                )
            ]
        ).send()

        await cl.Message(content="### 📱 Text Version\n" + "-"*20).send()
        await cl.Message(content=f"**Research:**\n{outputs['Research']}").send()
        await cl.Message(content=f"**Script:**\n{outputs['Script']}").send()
        await cl.Message(content=f"**Strategy:**\n{outputs['Strategy']}").send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()

