import os
import huggingface_hub
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel(), max_steps=15,
        verbosity_level=2,additional_authorized_imports=["json", "bs4", "requests"],)

response=agent.run("calculate the average raiology backlog for the nhs")


print(response)