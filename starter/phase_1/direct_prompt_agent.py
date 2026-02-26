# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Build extra kwargs for dual-provider support
extra_kwargs = {}
if os.getenv("LLM_PROVIDER"):
    extra_kwargs["provider"] = os.getenv("LLM_PROVIDER")
if os.getenv("ANTHROPIC_API_KEY"):
    extra_kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent
direct_agent = DirectPromptAgent(openai_api_key, **extra_kwargs)
# Use direct_agent to send the prompt and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# The DirectPromptAgent uses the LLM's pre-trained knowledge to answer questions.
# Since no system prompt or external knowledge is provided, the agent relies entirely
# on what the language model learned during its training (its parametric knowledge).
print("Knowledge source: The agent used the LLM's pre-trained (parametric) knowledge to answer the prompt. No external knowledge or system prompt was provided.")
