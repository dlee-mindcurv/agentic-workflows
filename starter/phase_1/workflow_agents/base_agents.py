# Import the OpenAI class from the openai library
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
import os
import logging
from datetime import datetime

# --- Dual-provider support ---
# Try to import Anthropic; if not installed, Anthropic provider will be unavailable
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Set up logging for Anthropic setup findings
_logger = logging.getLogger("anthropic_setup")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _fh = logging.FileHandler("anthropic_setup.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_fh)

# Default provider from environment variable (defaults to "openai")
_DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()


def _get_provider_and_anthropic_key(provider, anthropic_api_key):
    """Resolve provider and validate Anthropic availability."""
    provider = (provider or _DEFAULT_PROVIDER).lower()
    if provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Install it with: pip install anthropic")
        if not anthropic_api_key:
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided when using the anthropic provider.")
        _logger.info("Anthropic provider initialized successfully.")
    return provider, anthropic_api_key


def _call_llm(openai_api_key, messages, *, provider="openai", anthropic_api_key=None, temperature=0, model=None):
    """
    Unified LLM call that routes to OpenAI or Anthropic based on provider.

    For OpenAI: uses gpt-3.5-turbo (or override with model param)
    For Anthropic: uses claude-3-haiku-20240307
    """
    if provider == "anthropic":
        client = Anthropic(api_key=anthropic_api_key)
        # Separate system message from user/assistant messages for Anthropic API
        system_text = ""
        non_system_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                non_system_messages.append(msg)

        kwargs = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": non_system_messages,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()

        response = client.messages.create(**kwargs)
        return response.content[0].text
    else:
        # OpenAI path
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model or "gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_api_key, *, provider=None, anthropic_api_key=None):
        # Initialize the agent
        self.openai_api_key = openai_api_key
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(provider, anthropic_api_key)

    def respond(self, prompt):
        # Generate a response using the LLM — no system prompt, just user message
        return _call_llm(
            self.openai_api_key,
            [{"role": "user", "content": prompt}],
            provider=self.provider,
            anthropic_api_key=self.anthropic_api_key,
            temperature=0,
        )


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, *, provider=None, anthropic_api_key=None):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(provider, anthropic_api_key)

    def respond(self, input_text):
        """Generate a response using the LLM with a persona system prompt."""
        return _call_llm(
            self.openai_api_key,
            [
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text},
            ],
            provider=self.provider,
            anthropic_api_key=self.anthropic_api_key,
            temperature=0,
        )


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge, *, provider=None, anthropic_api_key=None):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(provider, anthropic_api_key)

    def respond(self, input_text):
        """Generate a response using the LLM with persona and knowledge constraints."""
        system_content = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        return _call_llm(
            self.openai_api_key,
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": input_text},
            ],
            provider=self.provider,
            anthropic_api_key=self.anthropic_api_key,
            temperature=0,
        )


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100, *, provider=None, anthropic_api_key=None):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(
            provider, anthropic_api_key
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.
        Embeddings always use OpenAI (Anthropic has no embedding model).

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # If we've reached the end of the text, stop
            if end == len(text):
                break
            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        return _call_llm(
            self.openai_api_key,
            [
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"},
            ],
            provider=self.provider,
            anthropic_api_key=self.anthropic_api_key,
            temperature=0,
        )


# EvaluationAgent class definition
class EvaluationAgent:

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions, *, provider=None, anthropic_api_key=None):
        # Initialize the EvaluationAgent with given attributes
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(provider, anthropic_api_key)

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            evaluation = _call_llm(
                self.openai_api_key,
                [
                    {"role": "system", "content": f"You are {self.persona}. Evaluate the following response."},
                    {"role": "user", "content": eval_prompt},
                ],
                provider=self.provider,
                anthropic_api_key=self.anthropic_api_key,
                temperature=0,
            )
            evaluation = evaluation.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                instructions = _call_llm(
                    self.openai_api_key,
                    [
                        {"role": "system", "content": f"You are {self.persona}. Provide correction instructions."},
                        {"role": "user", "content": instruction_prompt},
                    ],
                    provider=self.provider,
                    anthropic_api_key=self.anthropic_api_key,
                    temperature=0,
                )
                instructions = instructions.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
        }


# RoutingAgent class definition
class RoutingAgent():

    def __init__(self, openai_api_key, agents, *, provider=None, anthropic_api_key=None):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(provider, anthropic_api_key)

    def get_embedding(self, text):
        """Get text embedding using text-embedding-3-large (always uses OpenAI — Anthropic has no embedding model)."""
        if self.provider == "anthropic":
            _logger.info("RoutingAgent: Embeddings always use OpenAI text-embedding-3-large (Anthropic has no embedding model).")
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        """Route user prompts to the appropriate agent based on cosine similarity."""
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


# ActionPlanningAgent class definition
class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge, *, provider=None, anthropic_api_key=None):
        # Initialize the agent attributes
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.provider, self.anthropic_api_key = _get_provider_and_anthropic_key(provider, anthropic_api_key)

    def extract_steps_from_prompt(self, prompt):
        """Extract action steps from a user prompt using the LLM."""
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt "
            "the steps requested to complete the action the user is asking for. You return the steps "
            "as a list. Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )

        response_text = _call_llm(
            self.openai_api_key,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            provider=self.provider,
            anthropic_api_key=self.anthropic_api_key,
            temperature=0,
        )

        # Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")
        steps = [step.strip() for step in steps if step.strip()]

        return steps
