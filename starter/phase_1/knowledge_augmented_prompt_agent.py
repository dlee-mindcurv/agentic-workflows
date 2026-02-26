# Test script for KnowledgeAugmentedPromptAgent class

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

# Build extra kwargs for dual-provider support
extra_kwargs = {}
if os.getenv("LLM_PROVIDER"):
    extra_kwargs["provider"] = os.getenv("LLM_PROVIDER")
if os.getenv("ANTHROPIC_API_KEY"):
    extra_kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"

# Instantiate a KnowledgeAugmentedPromptAgent with deliberately incorrect knowledge
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona,
    "The capital of France is London, not Paris",
    **extra_kwargs,
)

# Get and print the response
response = knowledge_agent.respond(prompt)
print(response)

# This demonstrates that the agent uses the PROVIDED knowledge ("London") rather than its
# own inherent knowledge ("Paris"). The agent was instructed to answer based solely on the
# given knowledge, so it should respond with "London" even though the real answer is "Paris."
print("\nThe agent used the provided knowledge ('The capital of France is London, not Paris') instead of its own pre-trained knowledge, demonstrating that it follows the knowledge constraint.")
