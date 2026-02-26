# Test script for AugmentedPromptAgent class

from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Build extra kwargs for dual-provider support
extra_kwargs = {}
if os.getenv("LLM_PROVIDER"):
    extra_kwargs["provider"] = os.getenv("LLM_PROVIDER")
if os.getenv("ANTHROPIC_API_KEY"):
    extra_kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona, **extra_kwargs)

# Send the prompt to the agent and store the response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# Knowledge source: The agent used the LLM's pre-trained (parametric) knowledge to answer
# the question about the capital of France.
# Persona impact: The system prompt specifying the persona "college professor" caused the
# agent to frame its response in an academic style, starting with "Dear students," which
# demonstrates how the persona shapes the tone and format of the response without changing
# the factual content.
print("\nKnowledge source: The agent used the LLM's pre-trained knowledge to answer the question.")
print("Persona impact: The system prompt persona caused the agent to respond as a college professor, starting with 'Dear students,' — shaping the tone without changing the factual content.")
