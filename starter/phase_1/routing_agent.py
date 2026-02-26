# Test script for RoutingAgent class

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Build extra kwargs for dual-provider support
extra_kwargs = {}
if os.getenv("LLM_PROVIDER"):
    extra_kwargs["provider"] = os.getenv("LLM_PROVIDER")
if os.getenv("ANTHROPIC_API_KEY"):
    extra_kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

persona = "You are a college professor"

# Define the Texas Knowledge Augmented Prompt Agent
knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge, **extra_kwargs)

# Define the Europe Knowledge Augmented Prompt Agent
knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge, **extra_kwargs)

# Define the Math Knowledge Augmented Prompt Agent
persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge, **extra_kwargs)

routing_agent = RoutingAgent(openai_api_key, {}, **extra_kwargs)
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x),
    },
]

routing_agent.agents = agents

# Test routing with different prompts
print("=== Prompt 1: Tell me about the history of Rome, Texas ===")
response1 = routing_agent.route("Tell me about the history of Rome, Texas")
print(f"Response: {response1}\n")

print("=== Prompt 2: Tell me about the history of Rome, Italy ===")
response2 = routing_agent.route("Tell me about the history of Rome, Italy")
print(f"Response: {response2}\n")

print("=== Prompt 3: One story takes 2 days, and there are 20 stories ===")
response3 = routing_agent.route("One story takes 2 days, and there are 20 stories")
print(f"Response: {response3}\n")
