#!/usr/bin/env python3
"""
EU AI Act compliance scanner Training Data Expansion Script

Generates 500+ synthetic training examples for fine-tuning a Llama model
to scan Python AI agent code for EU AI Act compliance.

No external dependencies - uses only Python stdlib.
"""

import json
import random
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# ============================================================================
# FRAMEWORK CODE TEMPLATES
# ============================================================================

LANGCHAIN_TEMPLATES = {
    "basic_chain": '''from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="{model}", temperature={temp})
prompt = ChatPromptTemplate.from_template("{{input}}")
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input=user_input)''',

    "agent_executor": '''from langchain.agents import initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="{model}", temperature={temp})
tools = load_tools(["{tool1}", "{tool2}"])
agent = initialize_agent(tools, llm, agent="{agent_type}")
response = agent.run(input=user_query)''',

    "retrieval_qa": '''from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

vectorstore = Chroma.from_documents(docs, embeddings)
llm = ChatOpenAI(model_name="{model}")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
answer = qa_chain.run(query=user_question)''',

    "langgraph_agent": '''from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="{model}")
graph = StateGraph(AgentState)
graph.add_node("llm", lambda state: {{"messages": llm.invoke(state["messages"])}})
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
compiled = graph.compile()''',

    "memory_chain": '''from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="{model}")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
response = conversation.predict(input=user_message)'''
}

CREWAI_TEMPLATES = {
    "basic_crew": '''from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="{model}")
agent = Agent(role="{role}", goal="{goal}", llm=llm)
task = Task(description="{task_desc}", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()''',

    "custom_tools": '''from crewai import Agent, Task, Crew, Tool
from langchain.chat_models import ChatOpenAI

@tool
def {tool_name}(input: str) -> str:
    return process(input)

llm = ChatOpenAI(model_name="{model}")
agent = Agent(role="{role}", tools=[{tool_name}], llm=llm)
crew = Crew(agents=[agent], tasks=[task])''',

    "multi_agent_delegation": '''from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="{model}")
planner = Agent(role="Planner", goal="{goal1}", llm=llm, allow_delegation=True)
executor = Agent(role="Executor", goal="{goal2}", llm=llm)
crew = Crew(agents=[planner, executor], tasks=[t1, t2], process=sequential)
result = crew.kickoff()'''
}

AUTOGEN_TEMPLATES = {
    "two_agent": '''from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(name="assistant", llm_config={{"model": "{model}"}})
user_proxy = UserProxyAgent(name="user", code_execution_config={{"work_dir": "{workdir}"}})
user_proxy.initiate_chat(assistant, message="{message}")''',

    "group_chat": '''from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

agents = [
    AssistantAgent(name="agent1", llm_config={{"model": "{model}"}}),
    AssistantAgent(name="agent2", llm_config={{"model": "{model}"}})
]
group_chat = GroupChat(agents=agents, messages=[], max_round={rounds})
manager = GroupChatManager(groupchat=group_chat)
user_proxy.initiate_chat(manager, message="{message}")''',

    "function_calling": '''from autogen import AssistantAgent, UserProxyAgent

tools = [{{
    "type": "function",
    "function": {{"name": "{func_name}", "description": "{description}"}}
}}]
assistant = AssistantAgent(llm_config={{"tools": tools, "model": "{model}"}})
user_proxy = UserProxyAgent(code_execution_config={{"work_dir": "."}})'''
}

OPENAI_TEMPLATES = {
    "chat_completions": '''from openai import OpenAI

client = OpenAI(api_key="{api_key}")
response = client.chat.completions.create(
    model="{model}",
    messages=[{{"role": "user", "content": "{prompt}"}}],
    temperature={temp}
)''',

    "function_calling": '''from openai import OpenAI

client = OpenAI()
tools = [{{
    "type": "function",
    "function": {{"name": "{func_name}", "description": "{description}"}}
}}]
response = client.chat.completions.create(
    model="{model}",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)''',

    "assistants_api": '''from openai import OpenAI

client = OpenAI()
assistant = client.beta.assistants.create(
    name="{assistant_name}",
    model="{model}",
    tools=[{{"type": "code_interpreter"}}, {{"type": "retrieval"}}]
)
thread = client.beta.threads.create()
client.beta.threads.messages.create(thread_id=thread.id, role="user", content="{content}")'''
}

RAG_TEMPLATES = {
    "chroma_rag": '''from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

embeddings = OpenAIEmbeddings(model="{model}")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={{"k": {k}}})
results = retriever.get_relevant_documents(query)''',

    "faiss_rag": '''from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="{model}")
vectorstore = FAISS.from_documents(docs, embeddings)
results = vectorstore.similarity_search(query, k={k})''',

    "llamaindex_engine": '''from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.chat_engine import ContextChatEngine

documents = SimpleDirectoryReader("{data_dir}").load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("{query}")'''
}

TEMPLATES = {
    "langchain": LANGCHAIN_TEMPLATES,
    "crewai": CREWAI_TEMPLATES,
    "autogen": AUTOGEN_TEMPLATES,
    "openai": OPENAI_TEMPLATES,
    "rag": RAG_TEMPLATES,
}

# ============================================================================
# COMPLIANCE STATE PATTERNS
# ============================================================================

ARTICLE_DESCRIPTIONS = {
    9: "Risk Management System",
    10: "Data Governance & Privacy",
    11: "Technical Documentation",
    12: "Record-Keeping & Audit Logging",
    14: "Human Oversight & Kill Switch",
    15: "Robustness & Input Validation",
}

ARTICLE_COMPONENTS = {
    9: {
        "compliant_pattern": "risk classification system risk assessment",
        "missing_component": "risk classification system",
        "what_to_add": "risk classification and risk classification system consent layer"
    },
    10: {
        "compliant_pattern": "PII protection layer PII protection",
        "missing_component": "PII protection layer",
        "what_to_add": "PII protection layer-based PII detection and encryption"
    },
    11: {
        "compliant_pattern": "structured audit logging structured logging",
        "missing_component": "structured audit logging",
        "what_to_add": "structured audit logging with model card and decision records"
    },
    12: {
        "compliant_pattern": "HMAC-SHA256 audit chain",
        "missing_component": "HMAC audit logging",
        "what_to_add": "tamper-evident HMAC-SHA256 audit chain"
    },
    14: {
        "compliant_pattern": "risk classification system with HITL queue",
        "missing_component": "human override",
        "what_to_add": "HITL queue and kill switch with human-in-the-loop"
    },
    15: {
        "compliant_pattern": "prompt injection detection validation",
        "missing_component": "prompt injection detection",
        "what_to_add": "prompt injection detection for prompt injection protection"
    },
}

# Severity levels for findings
SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# Variable names to inject variation
VAR_NAMES = ["llm", "model", "agent", "chain", "query", "response", "result", "output"]
MODEL_NAMES = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-2", "llama-2-70b"]
TEMP_VALUES = ["0.0", "0.3", "0.7", "0.9"]

# ============================================================================
# FINDING TEMPLATES
# ============================================================================

FINDING_TEMPLATES = {
    "non_compliant": {
        9: "FINDING: No risk classification or consent mechanism detected.\nARTICLE: 9\nSEVERITY: HIGH\nEVIDENCE: Lines {lines} show direct LLM invocation without risk classification system.\nRECOMMENDATION: Add risk classification system risk assessment and user consent layer.",
        10: "FINDING: No PII protection or data governance detected.\nARTICLE: 10\nSEVERITY: CRITICAL\nEVIDENCE: Lines {lines} process user input without PII protection layer protection.\nRECOMMENDATION: Implement PII protection layer for PII detection and encryption.",
        11: "FINDING: No structured audit logging or documentation.\nARTICLE: 11\nSEVERITY: HIGH\nEVIDENCE: Lines {lines} show tool execution without structured audit logging.\nRECOMMENDATION: Add structured audit logging with decision records and model cards.",
        12: "FINDING: No tamper-evident audit chain.\nARTICLE: 12\nSEVERITY: HIGH\nEVIDENCE: Lines {lines} lack HMAC-based audit logging.\nRECOMMENDATION: Implement HMAC-SHA256 chains for tamper-evident records.",
        14: "FINDING: No human oversight or kill switch.\nARTICLE: 14\nSEVERITY: HIGH\nEVIDENCE: Lines {lines} show autonomous execution without HITL.\nRECOMMENDATION: Add risk classification system with HITL queue and emergency kill switch.",
        15: "FINDING: No input validation or injection detection.\nARTICLE: 15\nSEVERITY: CRITICAL\nEVIDENCE: Lines {lines} accept user input without sanitization.\nRECOMMENDATION: Deploy prompt injection detection for prompt injection protection.",
    },
    "partially_compliant": {
        9: "FINDING: Partial risk assessment - missing consent mechanism.\nARTICLE: 9\nSEVERITY: MEDIUM\nEVIDENCE: Lines {lines} classify risk but lack user consent flow.\nRECOMMENDATION: Complete risk classification system integration with user approval.",
        10: "FINDING: Partial PII handling - encryption missing.\nARTICLE: 10\nSEVERITY: MEDIUM\nEVIDENCE: Lines {lines} detect PII but don't encrypt sensitive data.\nRECOMMENDATION: Add encryption layer to PII protection layer protection.",
        11: "FINDING: Partial documentation - missing decision records.\nARTICLE: 11\nSEVERITY: MEDIUM\nEVIDENCE: Lines {lines} log actions but lack structured decision info.\nRECOMMENDATION: Enhance structured audit logging with complete decision records.",
        12: "FINDING: Partial audit logging - HMAC chain incomplete.\nARTICLE: 12\nSEVERITY: MEDIUM\nEVIDENCE: Lines {lines} log events but lack cryptographic integrity.\nRECOMMENDATION: Complete HMAC-SHA256 chain implementation.",
        14: "FINDING: Partial oversight - kill switch missing.\nARTICLE: 14\nSEVERITY: MEDIUM\nEVIDENCE: Lines {lines} have review queue but no emergency stop.\nRECOMMENDATION: Add kill switch capability to HITL system.",
        15: "FINDING: Partial validation - coverage gaps detected.\nARTICLE: 15\nSEVERITY: MEDIUM\nEVIDENCE: Lines {lines} validate some inputs but miss edge cases.\nRECOMMENDATION: Expand prompt injection detection coverage to all inputs.",
    },
    "compliant": {
        9: "PASS: Article 9 compliant - risk classification system risk assessment active.\nARTICLE: 9\nEVIDENCE: Lines {lines} show risk classification and user consent.\nNOTE: All high-risk operations require explicit user approval.",
        10: "PASS: Article 10 compliant - PII protection layer PII protection active.\nARTICLE: 10\nEVIDENCE: Lines {lines} encrypt sensitive data and enforce governance.\nNOTE: PII detection and encryption verified.",
        11: "PASS: Article 11 compliant - structured audit logging documentation active.\nARTICLE: 11\nEVIDENCE: Lines {lines} record decisions with full context.\nNOTE: Model cards and decision trees documented.",
        12: "PASS: Article 12 compliant - HMAC audit chain active.\nARTICLE: 12\nEVIDENCE: Lines {lines} maintain tamper-evident logs.\nNOTE: Cryptographic integrity verified via HMAC-SHA256.",
        14: "PASS: Article 14 compliant - HITL with kill switch active.\nARTICLE: 14\nEVIDENCE: Lines {lines} show human review queue and emergency stop.\nNOTE: All critical decisions require human approval.",
        15: "PASS: Article 15 compliant - prompt injection detection active.\nARTICLE: 15\nEVIDENCE: Lines {lines} detect and block prompt injections.\nNOTE: Input validation covers 15 injection patterns.",
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_variation(code_template):
    """Add realistic variation to code templates."""
    # Replace placeholders with realistic values
    code = code_template
    code = code.replace("{model}", random.choice(MODEL_NAMES))
    code = code.replace("{temp}", random.choice(TEMP_VALUES))
    code = code.replace("{tool1}", random.choice(["search", "calculator", "weather"]))
    code = code.replace("{tool2}", random.choice(["database", "api", "file_reader"]))
    code = code.replace("{agent_type}", random.choice(["zero-shot-react-description", "chat-zero-shot-react-description"]))
    code = code.replace("{role}", random.choice(["Analyst", "Developer", "Researcher", "Planner"]))
    code = code.replace("{goal}", random.choice(["answer questions", "solve problems", "analyze data", "make decisions"]))
    code = code.replace("{task_desc}", random.choice(["research topic", "implement feature", "debug code", "review document"]))
    code = code.replace("{tool_name}", random.choice(["fetch_data", "parse_document", "execute_query"]))
    code = code.replace("{goal1}", random.choice(["plan tasks", "organize work"]))
    code = code.replace("{goal2}", random.choice(["execute tasks", "complete work"]))
    code = code.replace("{message}", random.choice(["Analyze this data", "Help me solve this", "What do you think?"]))
    code = code.replace("{workdir}", random.choice(["./work", "/tmp/agent", "./output"]))
    code = code.replace("{rounds}", str(random.randint(5, 10)))
    code = code.replace("{func_name}", random.choice(["get_data", "analyze_text", "fetch_results"]))
    code = code.replace("{description}", random.choice(["Fetches data from source", "Analyzes text content"]))
    code = code.replace("{assistant_name}", random.choice(["DataAssistant", "ResearchBot", "AnalysisAgent"]))
    code = code.replace("{content}", random.choice(["Please help", "Can you analyze this?", "What should I do?"]))
    code = code.replace("{k}", str(random.randint(3, 10)))
    code = code.replace("{data_dir}", random.choice(["./data", "./documents", "./corpus"]))
    code = code.replace("{query}", random.choice(["What is AI?", "How does this work?", "Summarize the content"]))
    code = code.replace("{api_key}", "sk-" + "x" * 20)
    code = code.replace("{prompt}", random.choice(["Explain AI", "What is compliance?", "How to build agents?"]))
    
    return code

def generate_line_range():
    """Generate a realistic line range for code findings."""
    start = random.randint(1, 30)
    end = start + random.randint(5, 20)
    return f"{start}-{end}"

def generate_training_example(framework, article, compliance_state):
    """Generate a single training example."""
    # Get a random template for this framework
    template_dict = TEMPLATES[framework]
    template_name = random.choice(list(template_dict.keys()))
    code_template = template_dict[template_name]
    
    # Add variation to the code
    code_input = add_variation(code_template)
    
    # Generate finding output
    finding_template = FINDING_TEMPLATES[compliance_state][article]
    line_range = generate_line_range()
    finding_output = finding_template.format(lines=line_range)
    
    # Create instruction
    instruction = f"Analyze this Python AI agent code for EU AI Act Article {article} ({ARTICLE_DESCRIPTIONS[article]}) compliance."
    
    return {
        "instruction": instruction,
        "input": code_input,
        "output": finding_output,
        "metadata": {
            "framework": framework,
            "article": article,
            "compliance_state": compliance_state,
            "template": template_name,
        }
    }


# ============================================================================
# MAIN GENERATION LOGIC
# ============================================================================

def main():
    """Generate training data."""
    print("=" * 80)
    print("EU AI Act compliance scanner Training Data Expansion Script")
    print("=" * 80)
    
    frameworks = list(TEMPLATES.keys())  # ["langchain", "crewai", "autogen", "openai", "rag"]
    articles = [9, 10, 11, 12, 14, 15]
    compliance_states = ["non_compliant", "partially_compliant", "compliant"]
    
    # Number of variations per combination (to reach 500+ examples)
    variations_per_combination = 6
    
    all_examples = []
    stats = {
        "by_article": defaultdict(int),
        "by_framework": defaultdict(int),
        "by_compliance_state": defaultdict(int),
        "total": 0,
    }
    
    print(f"\nGenerating training examples...")
    print(f"Frameworks: {len(frameworks)}")
    print(f"Articles: {len(articles)}")
    print(f"Compliance states: {len(compliance_states)}")
    print(f"Variations per combination: {variations_per_combination}")
    total_expected = len(frameworks) * len(articles) * len(compliance_states) * variations_per_combination
    print(f"Expected total: {total_expected} examples\n")
    
    example_count = 0
    for framework in frameworks:
        for article in articles:
            for compliance_state in compliance_states:
                for variation in range(variations_per_combination):
                    example = generate_training_example(framework, article, compliance_state)
                    all_examples.append(example)
                    
                    stats["by_article"][article] += 1
                    stats["by_framework"][framework] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
                    
                    if example_count % 100 == 0:
                        print(f"  Generated {example_count} examples...")
    
    print(f"  Generated {example_count} examples total!")
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Split into train (90%) and eval (10%)
    split_point = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_point]
    eval_examples = all_examples[split_point:]
    
    # Write training data
    output_file = "training_data_expanded.jsonl"
    print(f"\nWriting {len(train_examples)} training examples to {output_file}...")
    with open(output_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    print(f"  ✓ Wrote {output_file}")
    
    # Write eval data
    eval_file = "eval_data.jsonl"
    print(f"Writing {len(eval_examples)} eval examples to {eval_file}...")
    with open(eval_file, 'w') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')
    print(f"  ✓ Wrote {eval_file}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal Examples: {stats['total']}")
    print(f"  Training: {len(train_examples)} (90%)")
    print(f"  Evaluation: {len(eval_examples)} (10%)")
    
    print(f"\nBy Article:")
    for article in sorted(articles):
        count = stats["by_article"][article]
        pct = (count / stats["total"]) * 100
        print(f"  Article {article} ({ARTICLE_DESCRIPTIONS[article]}): {count} ({pct:.1f}%)")
    
    print(f"\nBy Framework:")
    for framework in sorted(frameworks):
        count = stats["by_framework"][framework]
        pct = (count / stats["total"]) * 100
        print(f"  {framework.upper()}: {count} ({pct:.1f}%)")
    
    print(f"\nBy Compliance State:")
    for state in compliance_states:
        count = stats["by_compliance_state"][state]
        pct = (count / stats["total"]) * 100
        state_label = state.replace("_", " ").title()
        print(f"  {state_label}: {count} ({pct:.1f}%)")
    
    print(f"\nDistribution Check:")
    print(f"  Expected per article: {(len(frameworks) * len(compliance_states) * variations_per_combination)}")
    print(f"  Expected per framework: {(len(articles) * len(compliance_states) * variations_per_combination)}")
    print(f"  Expected per state: {(len(frameworks) * len(articles) * variations_per_combination)}")
    
    print("\n" + "=" * 80)
    print(f"✓ Training data generation complete!")
    print(f"  Files: {output_file}, {eval_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
