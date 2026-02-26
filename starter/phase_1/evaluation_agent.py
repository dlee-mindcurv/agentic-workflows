# Test script for EvaluationAgent class

from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Build extra kwargs for dual-provider support
extra_kwargs = {}
if os.getenv("LLM_PROVIDER"):
    extra_kwargs["provider"] = os.getenv("LLM_PROVIDER")
if os.getenv("ANTHROPIC_API_KEY"):
    extra_kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge, **extra_kwargs)

# Parameters for the Evaluation Agent
eval_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
    openai_api_key,
    eval_persona,
    evaluation_criteria,
    knowledge_agent,
    10,
    **extra_kwargs,
)

# Evaluate the prompt and print the response from the EvaluationAgent
result = evaluation_agent.evaluate(prompt)
print(f"\nFinal Response: {result['final_response']}")
print(f"Final Evaluation: {result['evaluation']}")
print(f"Total Iterations: {result['iterations']}")
