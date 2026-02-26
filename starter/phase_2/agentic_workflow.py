# agentic_workflow.py

# TODO 1: Import the required agents from workflow_agents.base_agents
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# TODO 2: Load the OpenAI key into a variable called openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Build extra kwargs for dual-provider support
extra_kwargs = {}
if os.getenv("LLM_PROVIDER"):
    extra_kwargs["provider"] = os.getenv("LLM_PROVIDER")
if os.getenv("ANTHROPIC_API_KEY"):
    extra_kwargs["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")

# TODO 3: Load the product spec document into a variable called product_spec
with open("Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO 4: Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning, **extra_kwargs)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO 5: Complete this knowledge string by appending the product_spec
    + product_spec
)
# TODO 6: Instantiate a product_manager_knowledge_agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager, **extra_kwargs
)

# Product Manager - Evaluation Agent
# TODO 7: Define the persona and evaluation criteria for a Product Manager evaluation agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_pm = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_pm,
    product_manager_knowledge_agent,
    10,
    **extra_kwargs,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager, **extra_kwargs
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO 8: Instantiate a program_manager_evaluation_agent
evaluation_criteria_pgm = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_pgm,
    program_manager_knowledge_agent,
    10,
    **extra_kwargs,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer, **extra_kwargs
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO 9: Instantiate a development_engineer_evaluation_agent
evaluation_criteria_dev = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev,
    development_engineer_knowledge_agent,
    10,
    **extra_kwargs,
)


# TODO 11: Define the support functions for the routes of the routing agent
# (Must be defined BEFORE the RoutingAgent instantiation that references them)
def product_manager_support_function(input_query):
    """Get user stories from PM knowledge agent, then validate with evaluation agent."""
    response = product_manager_knowledge_agent.respond(input_query)
    result = product_manager_evaluation_agent.evaluate(response)
    return result["final_response"]


def program_manager_support_function(input_query):
    """Get product features from PgM knowledge agent, then validate with evaluation agent."""
    response = program_manager_knowledge_agent.respond(input_query)
    result = program_manager_evaluation_agent.evaluate(response)
    return result["final_response"]


def development_engineer_support_function(input_query):
    """Get engineering tasks from Dev knowledge agent, then validate with evaluation agent."""
    response = development_engineer_knowledge_agent.respond(input_query)
    result = development_engineer_evaluation_agent.evaluate(response)
    return result["final_response"]


# Routing Agent
# TODO 10: Instantiate a routing_agent with routes for PM, PgM, and Dev Engineer
routes = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only based on the product specification.",
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining and organizing product features by grouping related user stories.",
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining detailed engineering and development tasks to build and implement user stories.",
        "func": development_engineer_support_function,
    },
]
routing_agent = RoutingAgent(openai_api_key, routes, **extra_kwargs)


# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
workflow_prompt = "What would the development tasks for this product be?"
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

# TODO 12: Implement the workflow
# 1. Use the action_planning_agent to extract steps from the workflow_prompt
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"\nWorkflow steps extracted ({len(workflow_steps)} steps):")
for idx, step in enumerate(workflow_steps):
    print(f"  Step {idx+1}: {step}")

# 2. Initialize an empty list to store completed_steps
completed_steps = []

# 3. Loop through the extracted workflow steps
for idx, step in enumerate(workflow_steps):
    print(f"\n{'='*60}")
    print(f"Executing Step {idx+1}: {step}")
    print(f"{'='*60}")

    # Route the step to the appropriate support function
    result = routing_agent.route(step)
    completed_steps.append(result)

    print(f"\nResult for Step {idx+1}:\n{result}")

# 4. Print the final output of the workflow
print(f"\n--- Final Workflow Output ---\n")
print(f"Total steps completed: {len(completed_steps)}\n")
for idx, step_result in enumerate(completed_steps):
    print(f"\n{'='*60}")
    print(f"Step {idx+1} Result:")
    print(f"{'='*60}")
    print(step_result)

print(f"\n{'='*60}")
print("*** Workflow execution completed ***")
print(f"{'='*60}")
