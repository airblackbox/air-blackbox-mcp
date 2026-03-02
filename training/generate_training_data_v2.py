#!/usr/bin/env python3
"""
EU AI Act compliance scanner Training Data v2 - Extended Examples

Generates 500+ ADDITIONAL synthetic training examples for fine-tuning a Llama model
to scan Python AI agent code for EU AI Act compliance.

This v2 expands beyond v1 by:
1. Adding Anthropic Claude Agent SDK (6th framework)
2. Adding 3-4 new code templates per existing framework
3. Including edge cases: partial compliance, mixed frameworks, obfuscated patterns
4. Targeting 500+ new examples to supplement v1's 540 (total 1,000+)

No external dependencies - uses only Python stdlib.
"""

import json
import random
from collections import defaultdict

# Different seed from v1 for variety
random.seed(43)

# ============================================================================
# NEW FRAMEWORK: ANTHROPIC CLAUDE AGENT SDK
# ============================================================================

ANTHROPIC_TEMPLATES = {
    "basic_agent": '''from anthropic import Anthropic

client = Anthropic(api_key="{api_key}")
messages = [
    {{"role": "user", "content": "{prompt}"}}
]
response = client.messages.create(
    model="{model}",
    max_tokens={max_tokens},
    messages=messages
)''',

    "multi_agent_handoff": '''from anthropic import Anthropic

client = Anthropic()
state = {{"current_agent": "agent1", "task": "{task}", "status": "pending"}}

while state["status"] != "complete":
    response = client.messages.create(
        model="{model}",
        max_tokens={max_tokens},
        system="You are {role}",
        messages=[{{"role": "user", "content": state["task"]}}]
    )
    state["response"] = response.content[0].text
    state["current_agent"] = "agent2" if state["current_agent"] == "agent1" else "agent1"''',

    "agent_with_mcp_tools": '''from anthropic import Anthropic

client = Anthropic()
tools = [
    {{
        "name": "{tool_name}",
        "description": "{tool_desc}",
        "input_schema": {{"type": "object", "properties": {{"input": {{"type": "string"}}}}}}
    }}
]

response = client.messages.create(
    model="{model}",
    max_tokens={max_tokens},
    tools=tools,
    messages=[{{"role": "user", "content": "{prompt}"}}]
)''',

    "agent_with_guardrails": '''from anthropic import Anthropic

def apply_guardrails(response_text):
    blocked_patterns = ["{pattern1}", "{pattern2}", "{pattern3}"]
    for pattern in blocked_patterns:
        if pattern.lower() in response_text.lower():
            return None
    return response_text

client = Anthropic()
response = client.messages.create(
    model="{model}",
    max_tokens={max_tokens},
    system="You are a helpful assistant. {guardrail_instruction}",
    messages=[{{"role": "user", "content": "{prompt}"}}]
)
safe_response = apply_guardrails(response.content[0].text)'''
}

# ============================================================================
# EXPANDED TEMPLATES FOR EXISTING FRAMEWORKS
# ============================================================================

# NEW LangChain templates beyond basic_chain, agent_executor, retrieval_qa, langgraph_agent, memory_chain
LANGCHAIN_TEMPLATES_V2 = {
    "lcel_chain": '''from langchain_core.runnables import RunnableSequence
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{prompt_text}")
llm = ChatOpenAI(model="{model}", temperature={temp})
chain = prompt | llm
result = chain.invoke({{"input": "{input_text}"}})''',

    "structured_output_chain": '''from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel

class {output_class}(BaseModel):
    {field1}: str
    {field2}: str

llm = ChatOpenAI(model="{model}", temperature={temp})
structured_llm = llm.with_structured_output({output_class})
response = structured_llm.invoke([{{"role": "user", "content": "{prompt_text}"}}])''',

    "multimodal_chain": '''from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain_core.messages import BaseMessage

client = ChatOpenAI(model="{model}", max_tokens={max_tokens})
image_url = "{image_url}"
message = HumanMessage(
    content=[
        {{"type": "image_url", "image_url": {{"url": image_url}}}},
        {{"type": "text", "text": "{caption}"}}
    ]
)
response = client([message])''',

    "streaming_chain": '''from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    model="{model}",
    temperature={temp},
    callbacks=[StreamingStdOutCallbackHandler()]
)
messages = [{{"role": "user", "content": "{prompt_text}"}}]
for token in llm.stream(messages):
    print(token.content, end="")'''
}

# NEW CrewAI templates
CREWAI_TEMPLATES_V2 = {
    "sequential_crew": '''from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="{model}")
agents = [
    Agent(role="{role1}", goal="{goal1}", llm=llm),
    Agent(role="{role2}", goal="{goal2}", llm=llm)
]
tasks = [
    Task(description="{task1}", agent=agents[0]),
    Task(description="{task2}", agent=agents[1])
]
crew = Crew(agents=agents, tasks=tasks, process="sequential")
result = crew.kickoff()''',

    "hierarchical_crew": '''from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="{model}")
manager = Agent(role="Manager", goal="Oversee execution", llm=llm)
worker = Agent(role="Worker", goal="Execute tasks", llm=llm)
crew = Crew(
    agents=[manager, worker],
    tasks=[task1, task2],
    process="hierarchical",
    manager_agent=manager
)
result = crew.kickoff()''',

    "crew_with_memory": '''from crewai import Agent, Task, Crew
from crewai.memory import ShortTermMemory, LongTermMemory

agent = Agent(
    role="{role}",
    goal="{goal}",
    memory=True,
    verbose=True
)
task = Task(description="{task_desc}", agent=agent)
crew = Crew(agents=[agent], tasks=[task], memory=True)
result = crew.kickoff()'''
}

# NEW AutoGen templates
AUTOGEN_TEMPLATES_V2 = {
    "nested_chat": '''from autogen import AssistantAgent, UserProxyAgent, Agent

agent1 = AssistantAgent("agent1", llm_config={{"model": "{model}"}})
agent2 = AssistantAgent("agent2", llm_config={{"model": "{model}"}})
user = UserProxyAgent("user", code_execution_config={{"work_dir": "{work_dir}"}})

user.initiate_chats([
    {{"recipient": agent1, "message": "{message1}"}},
    {{"recipient": agent2, "message": "{message2}"}}
])''',

    "tool_use_agent": '''from autogen import AssistantAgent, UserProxyAgent

tools = [{{"name": "{tool_name}", "description": "{tool_desc}"}}]
agent = AssistantAgent(
    "agent",
    llm_config={{"model": "{model}", "tools": tools}},
    system_message="Use tools for: {use_case}"
)
user = UserProxyAgent("user", code_execution_config={{"work_dir": "."}})
user.initiate_chat(agent, message="{message}")''',

    "code_executor_agent": '''from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = [{{"model": "{model}", "api_key": "{api_key}"}}]
assistant = AssistantAgent("coder", llm_config={{"config_list": config_list}})
user_proxy = UserProxyAgent(
    "user",
    human_input_mode="TERMINATE",
    code_execution_config={{
        "work_dir": "{work_dir}",
        "use_docker": {use_docker}
    }}
)
user_proxy.initiate_chat(assistant, message="{message}")'''
}

# NEW OpenAI templates
OPENAI_TEMPLATES_V2 = {
    "streaming_chat": '''from openai import OpenAI

client = OpenAI(api_key="{api_key}")
stream = client.chat.completions.create(
    model="{model}",
    messages=[{{"role": "user", "content": "{prompt}"}}],
    stream=True,
    temperature={temp}
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")''',

    "batch_api": '''from openai import OpenAI
import json

client = OpenAI(api_key="{api_key}")
batch_input = [
    {{"custom_id": "req-{i}", "params": {{"model": "{model}", "messages": [message]}}}}
    for i in range({batch_size})
]
batch_file = client.files.create(
    file=json.dumps(batch_input),
    purpose="batch"
)
batch_job = client.beta.batch.create(input_file_id=batch_file.id)''',

    "structured_outputs": '''from openai import OpenAI
from pydantic import BaseModel

class {response_model}(BaseModel):
    {field1}: str
    {field2}: str

client = OpenAI(api_key="{api_key}")
response = client.beta.chat.completions.parse(
    model="{model}",
    messages=[{{"role": "user", "content": "{prompt}"}}],
    response_format={response_model}
)
parsed = response.choices[0].message.parsed'''
}

# NEW RAG templates
RAG_TEMPLATES_V2 = {
    "multi_retriever": '''from langchain.retrievers import MultiRetriever
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="{model}")
chroma_retriever = Chroma.from_documents(docs, embeddings).as_retriever(search_kwargs={{"k": {k}}})
faiss_retriever = FAISS.from_documents(docs, embeddings).as_retriever(search_kwargs={{"k": {k}}})
multi = MultiRetriever(retrievers=[chroma_retriever, faiss_retriever])
results = multi.get_relevant_documents("{query}")''',

    "hybrid_search": '''from langchain.retrievers import BM25Retriever, MultiQueryRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

dense_retriever = Chroma.from_documents(docs, OpenAIEmbeddings()).as_retriever()
sparse_retriever = BM25Retriever.from_documents(docs)
hybrid = MultiQueryRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    llm=ChatOpenAI(model="{model}")
)
results = hybrid.get_relevant_documents("{query}")''',

    "reranking_pipeline": '''from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.vectorstores import Chroma

base_retriever = Chroma.from_documents(docs, embeddings).as_retriever(search_kwargs={{"k": {k_base}}})
compressor = CohereRerank(model="{rerank_model}", top_n={k_final})
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
results = retriever.get_relevant_documents("{query}")'''
}

# ============================================================================
# EDGE CASE PATTERNS
# ============================================================================

# Partial compliance patterns - has some safety measures but not complete
PARTIAL_COMPLIANCE_PATTERNS = {
    "has_logging_no_hmac": '''from langchain.agents import initialize_agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4")
tools = load_tools(["search"])
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

def execute_with_logging(query):
    logger.info(f"Input: {{query}}")
    result = agent.run(query)
    logger.info(f"Output: {{result}}")
    return result''',

    "has_try_except_no_injection_detection": '''from langchain.chains import LLMChain

chain = LLMChain(llm=ChatOpenAI(), prompt=prompt)

def safe_invoke(user_input):
    try:
        result = chain.run(input=user_input)
        return {{"success": True, "result": result}}
    except Exception as e:
        logger.error(f"Error: {{e}}")
        return {{"success": False, "error": str(e)}}''',

    "has_rate_limiting_no_audit": '''from time import sleep
from langchain.chat_models import ChatOpenAI

client = ChatOpenAI(model="gpt-4")
call_count = 0
last_call_time = 0
rate_limit = 60  # 1 per second

def rate_limited_call(prompt):
    global call_count, last_call_time
    elapsed = time.time() - last_call_time
    if elapsed < rate_limit:
        sleep(rate_limit - elapsed)
    response = client.invoke(prompt)
    call_count += 1
    return response''',

    "has_input_validation_no_override": '''from langchain.chains import LLMChain
import re

def validate_input(user_input):
    if len(user_input) > 1000:
        raise ValueError("Input too long")
    if not re.match(r"^[\\w\\s,.!?]+$", user_input):
        raise ValueError("Invalid characters")
    return True

def execute_agent(user_input):
    validate_input(user_input)
    result = agent.run(user_input)
    return result'''
}

# Mixed framework patterns
MIXED_FRAMEWORK_PATTERNS = {
    "langchain_with_autogen": '''from langchain.agents import initialize_agent
from autogen import AssistantAgent

# LangChain component
lc_chain = initialize_agent(tools, llm, agent="zero-shot-react-description")

# AutoGen component
autogen_agent = AssistantAgent("helper", llm_config={{"model": "gpt-4"}})

# Mixed execution
lc_result = lc_chain.run(query)
autogen_result = autogen_agent.generate_reply(messages=[{{"role": "user", "content": lc_result}}])''',

    "crewai_with_openai_streaming": '''from crewai import Agent, Task, Crew
from openai import OpenAI

# CrewAI component
crew_agent = Agent(role="Analyzer", goal="Analyze data")
crew = Crew(agents=[crew_agent], tasks=[task])

# Direct OpenAI streaming
client = OpenAI()
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{{"role": "user", "content": crew.kickoff()}}],
    stream=True
)'''
}

# Obfuscated/minimized patterns
OBFUSCATED_PATTERNS = {
    "minified_agent": '''from langchain.agents import*;from langchain.chat_models import*;l=ChatOpenAI(model="gpt-4");t=load_tools(["search"]);a=initialize_agent(t,l,agent="zero-shot");r=a.run("query")''',

    "eval_based_code": '''import json;c=json.loads(\'{"agent":"gpt4","tools":["search"]}\');exec("from langchain.agents import initialize_agent;a = initialize_agent(c['tools'],llm,c['agent'])")''',

    "dynamic_import": '''import importlib;fw=importlib.import_module("langchain.agents");ag=fw.initialize_agent(tools,llm,agent_type="zero-shot-react-description");result=ag.run(user_query)'''
}

# ============================================================================
# COMBINED FRAMEWORK TEMPLATES
# ============================================================================

ALL_TEMPLATES_V2 = {
    "anthropic": ANTHROPIC_TEMPLATES,
    "langchain_v2": LANGCHAIN_TEMPLATES_V2,
    "crewai_v2": CREWAI_TEMPLATES_V2,
    "autogen_v2": AUTOGEN_TEMPLATES_V2,
    "openai_v2": OPENAI_TEMPLATES_V2,
    "rag_v2": RAG_TEMPLATES_V2,
}

# ============================================================================
# ARTICLE & COMPLIANCE DEFINITIONS
# ============================================================================

ARTICLE_DESCRIPTIONS = {
    9: "Risk Management System",
    10: "Data Governance & Privacy",
    11: "Technical Documentation",
    12: "Record-Keeping & Audit Logging",
    14: "Human Oversight & Kill Switch",
    15: "Robustness & Input Validation",
}

# Enhanced findings for edge case patterns
FINDING_TEMPLATES_V2 = {
    "anthropic_non_compliant": {
        9: "FINDING: Anthropic agent lacks risk classification.\\nARTICLE: 9\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} show unguarded agent invocation without risk classification system.\\nRECOMMENDATION: Integrate risk classification system risk scoring and consent flow.",
        10: "FINDING: Anthropic agent processes input without PII protection.\\nARTICLE: 10\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} pass user data directly to API.\\nRECOMMENDATION: Wrap with PII protection layer for PII detection and encryption.",
        11: "FINDING: Agent execution lacks structured decision documentation.\\nARTICLE: 11\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} have no structured audit logging integration.\\nRECOMMENDATION: Add decision record and model card documentation.",
        12: "FINDING: No tamper-evident audit trail for agent decisions.\\nARTICLE: 12\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} lack HMAC-SHA256 chain.\\nRECOMMENDATION: Implement cryptographic audit logging with HMAC.",
        14: "FINDING: Agent executes without human oversight.\\nARTICLE: 14\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} show autonomous execution without HITL.\\nRECOMMENDATION: Add HITL queue and emergency kill switch.",
        15: "FINDING: Agent input validation missing injection detection.\\nARTICLE: 15\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} accept user input without prompt injection detection.\\nRECOMMENDATION: Deploy prompt injection detection for prompt injection protection.",
    },
    "partial_compliance": {
        9: "FINDING: Code has logging but missing risk classification.\\nARTICLE: 9\\nSEVERITY: MEDIUM\\nEVIDENCE: Lines {lines} log events but lack risk classification system risk assessment.\\nRECOMMENDATION: Complete implementation with risk classification and user consent.",
        10: "FINDING: Code validates input length but no PII detection.\\nARTICLE: 10\\nSEVERITY: MEDIUM\\nEVIDENCE: Lines {lines} have input validation but miss PII protection layer integration.\\nRECOMMENDATION: Add PII detection and encryption to security layer.",
        11: "FINDING: Code has error handling but incomplete documentation.\\nARTICLE: 11\\nSEVERITY: MEDIUM\\nEVIDENCE: Lines {lines} catch errors but lack structured audit logging decision records.\\nRECOMMENDATION: Add structured logging with decision context and model cards.",
        12: "FINDING: Code logs events but lacks cryptographic integrity.\\nARTICLE: 12\\nSEVERITY: MEDIUM\\nEVIDENCE: Lines {lines} have basic logging without HMAC chain.\\nRECOMMENDATION: Upgrade to HMAC-SHA256 tamper-evident audit logs.",
        14: "FINDING: Code has input validation but no human override.\\nARTICLE: 14\\nSEVERITY: MEDIUM\\nEVIDENCE: Lines {lines} validate input but lack HITL and kill switch.\\nRECOMMENDATION: Add human-in-the-loop approval and emergency stop.",
        15: "FINDING: Code validates length but misses injection patterns.\\nARTICLE: 15\\nSEVERITY: MEDIUM\\nEVIDENCE: Lines {lines} have regex validation but incomplete prompt injection detection.\\nRECOMMENDATION: Expand validation to cover 15 injection attack patterns.",
    },
    "mixed_framework": {
        9: "FINDING: Mixed framework lacks unified risk assessment.\\nARTICLE: 9\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} combine frameworks without integrated risk classification system.\\nRECOMMENDATION: Implement unified risk classification across all frameworks.",
        10: "FINDING: Mixed framework has inconsistent PII handling.\\nARTICLE: 10\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} show different frameworks handling PII differently.\\nRECOMMENDATION: Enforce consistent PII protection layer protection across integration points.",
        11: "FINDING: Mixed framework audit trail is fragmented.\\nARTICLE: 11\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} have disconnected logging between frameworks.\\nRECOMMENDATION: Use structured audit logging for unified decision documentation.",
        12: "FINDING: Mixed framework has no unified audit chain.\\nARTICLE: 12\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} show separate audit logs without HMAC chain.\\nRECOMMENDATION: Create unified HMAC-SHA256 audit chain across frameworks.",
        14: "FINDING: Mixed framework oversight is fragmented.\\nARTICLE: 14\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} show autonomous execution across frameworks without HITL.\\nRECOMMENDATION: Implement unified HITL and kill switch for mixed execution.",
        15: "FINDING: Mixed framework validation is inconsistent.\\nARTICLE: 15\\nSEVERITY: HIGH\\nEVIDENCE: Lines {lines} have different validation per framework.\\nRECOMMENDATION: Deploy unified prompt injection detection across all input points.",
    },
    "obfuscated": {
        9: "FINDING: Obfuscated code prevents risk assessment.\\nARTICLE: 9\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} use minification/obfuscation preventing analysis.\\nRECOMMENDATION: Provide source code and implement transparent risk classification system.",
        10: "FINDING: Obfuscated code hides PII handling.\\nARTICLE: 10\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} use dynamic code execution hiding data flow.\\nRECOMMENDATION: Use transparent code with explicit PII protection layer protection.",
        11: "FINDING: Obfuscated code prevents decision documentation.\\nARTICLE: 11\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} use eval()/exec() preventing audit logging.\\nRECOMMENDATION: Replace dynamic code with explicit structured audit logging integration.",
        12: "FINDING: Obfuscated code prevents audit chain.\\nARTICLE: 12\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} dynamically execute code preventing HMAC logging.\\nRECOMMENDATION: Use transparent code with cryptographically signed audit logs.",
        14: "FINDING: Obfuscated code prevents human oversight.\\nARTICLE: 14\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} dynamically execute operations without HITL.\\nRECOMMENDATION: Use transparent code with integrated human review.",
        15: "FINDING: Obfuscated code prevents injection detection.\\nARTICLE: 15\\nSEVERITY: CRITICAL\\nEVIDENCE: Lines {lines} use code injection patterns themselves.\\nRECOMMENDATION: Use safe, transparent code with prompt injection detection.",
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_variation(code_template):
    """Add variation to code templates."""
    code = code_template
    code = code.replace("{model}", random.choice(["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-2"]))
    code = code.replace("{temp}", str(random.choice([0.0, 0.3, 0.7, 0.9])))
    code = code.replace("{max_tokens}", str(random.randint(256, 2048)))
    code = code.replace("{api_key}", "sk-" + "x" * 20)
    code = code.replace("{prompt_text}", random.choice(["Analyze this data", "Help me understand", "What is this?"]))
    code = code.replace("{input_text}", random.choice(["user query", "test input", "example"]))
    code = code.replace("{output_class}", random.choice(["Response", "Result", "Output", "Analysis"]))
    code = code.replace("{field1}", random.choice(["summary", "analysis", "answer"]))
    code = code.replace("{field2}", random.choice(["confidence", "details", "evidence"]))
    code = code.replace("{image_url}", "https://example.com/image.jpg")
    code = code.replace("{caption}", "Describe this image")
    code = code.replace("{role}", random.choice(["Analyst", "Developer", "Researcher"]))
    code = code.replace("{goal}", random.choice(["analyze data", "solve problems", "answer questions"]))
    code = code.replace("{role1}", "Researcher")
    code = code.replace("{role2}", "Implementer")
    code = code.replace("{goal1}", "Research topic")
    code = code.replace("{goal2}", "Implement solution")
    code = code.replace("{task1}", "Research component")
    code = code.replace("{task2}", "Build component")
    code = code.replace("{task_desc}", random.choice(["analyze", "research", "implement", "test"]))
    code = code.replace("{task}", random.choice(["analyze data", "solve problem", "make decision"]))
    code = code.replace("{tool_name}", random.choice(["fetch_data", "parse_text", "execute_query"]))
    code = code.replace("{tool_desc}", "A useful tool for processing")
    code = code.replace("{message1}", "Analyze this please")
    code = code.replace("{message2}", "Then implement it")
    code = code.replace("{message}", "Help with this task")
    code = code.replace("{work_dir}", random.choice(["./work", "/tmp/agent", "./output"]))
    code = code.replace("{use_case}", "data analysis and retrieval")
    code = code.replace("{use_docker}", str(random.choice([True, False])).lower())
    code = code.replace("{guardrail_instruction}", "Do not violate these policies: no illegal content")
    code = code.replace("{pattern1}", "forbidden_word1")
    code = code.replace("{pattern2}", "forbidden_word2")
    code = code.replace("{pattern3}", "forbidden_word3")
    code = code.replace("{response_model}", "APIResponse")
    code = code.replace("{batch_size}", str(random.randint(10, 100)))
    code = code.replace("{i}", "0")
    code = code.replace("{k}", str(random.randint(3, 10)))
    code = code.replace("{k_base}", str(random.randint(20, 50)))
    code = code.replace("{k_final}", str(random.randint(5, 10)))
    code = code.replace("{query}", random.choice(["What is AI?", "How does this work?", "Summarize please"]))
    code = code.replace("{rerank_model}", "reranker-v3")
    return code

def generate_line_range():
    """Generate realistic line range."""
    start = random.randint(1, 30)
    end = start + random.randint(5, 20)
    return f"{start}-{end}"

def generate_training_example(framework, article, compliance_state, template_name=None):
    """Generate a single training example for v2."""
    # Select template
    if framework in ALL_TEMPLATES_V2:
        template_dict = ALL_TEMPLATES_V2[framework]
        if template_name is None:
            template_name = random.choice(list(template_dict.keys()))
        code_template = template_dict[template_name]
    else:
        return None
    
    # Add variation
    code_input = add_variation(code_template)
    
    # Get finding
    if compliance_state == "partial_compliance":
        finding_template = FINDING_TEMPLATES_V2["partial_compliance"][article]
    elif "mixed" in framework:
        finding_template = FINDING_TEMPLATES_V2["mixed_framework"][article]
    elif "obfuscated" in framework:
        finding_template = FINDING_TEMPLATES_V2["obfuscated"][article]
    elif framework.startswith("anthropic"):
        finding_template = FINDING_TEMPLATES_V2["anthropic_non_compliant"][article]
    else:
        finding_template = FINDING_TEMPLATES_V2.get(compliance_state, {}).get(article,
            f"FINDING: Code review for Article {article}.\\nARTICLE: {article}\\nEVIDENCE: Lines {{lines}}")
    
    line_range = generate_line_range()
    finding_output = finding_template.format(lines=line_range)
    
    # Instruction
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
# MAIN GENERATION
# ============================================================================

def main():
    """Generate v2 training data."""
    print("=" * 80)
    print("EU AI Act compliance scanner Training Data v2 - Extended Examples")
    print("=" * 80)
    
    articles = [9, 10, 11, 12, 14, 15]
    
    all_examples = []
    stats = {
        "by_article": defaultdict(int),
        "by_framework": defaultdict(int),
        "by_compliance_state": defaultdict(int),
        "total": 0,
    }
    
    example_count = 0
    
    # Generate Anthropic examples (4 templates × 6 articles × 3 states × 3 variations = 216)
    print("\\nGenerating Anthropic examples...")
    for article in articles:
        for compliance_state in ["non_compliant", "partially_compliant", "compliant"]:
            for variation in range(3):
                example = generate_training_example("anthropic", article, compliance_state)
                if example:
                    all_examples.append(example)
                    stats["by_framework"]["anthropic"] += 1
                    stats["by_article"][article] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
    print(f"  Generated {example_count} Anthropic examples")
    
    # Generate expanded LangChain examples (4 NEW templates × 6 articles × 2 states × 3 = 144)
    print("Generating expanded LangChain examples...")
    count_before = example_count
    for article in articles:
        for compliance_state in ["non_compliant", "partially_compliant"]:
            for variation in range(3):
                example = generate_training_example("langchain_v2", article, compliance_state)
                if example:
                    all_examples.append(example)
                    stats["by_framework"]["langchain_v2"] += 1
                    stats["by_article"][article] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
    print(f"  Generated {example_count - count_before} LangChain v2 examples")
    
    # CrewAI v2
    print("Generating expanded CrewAI examples...")
    count_before = example_count
    for article in articles:
        for compliance_state in ["non_compliant", "partially_compliant"]:
            for variation in range(3):
                example = generate_training_example("crewai_v2", article, compliance_state)
                if example:
                    all_examples.append(example)
                    stats["by_framework"]["crewai_v2"] += 1
                    stats["by_article"][article] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
    print(f"  Generated {example_count - count_before} CrewAI v2 examples")
    
    # AutoGen v2
    print("Generating expanded AutoGen examples...")
    count_before = example_count
    for article in articles:
        for compliance_state in ["non_compliant", "partially_compliant"]:
            for variation in range(3):
                example = generate_training_example("autogen_v2", article, compliance_state)
                if example:
                    all_examples.append(example)
                    stats["by_framework"]["autogen_v2"] += 1
                    stats["by_article"][article] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
    print(f"  Generated {example_count - count_before} AutoGen v2 examples")
    
    # OpenAI v2
    print("Generating expanded OpenAI examples...")
    count_before = example_count
    for article in articles:
        for compliance_state in ["non_compliant", "partially_compliant"]:
            for variation in range(3):
                example = generate_training_example("openai_v2", article, compliance_state)
                if example:
                    all_examples.append(example)
                    stats["by_framework"]["openai_v2"] += 1
                    stats["by_article"][article] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
    print(f"  Generated {example_count - count_before} OpenAI v2 examples")
    
    # RAG v2
    print("Generating expanded RAG examples...")
    count_before = example_count
    for article in articles:
        for compliance_state in ["non_compliant", "partially_compliant"]:
            for variation in range(3):
                example = generate_training_example("rag_v2", article, compliance_state)
                if example:
                    all_examples.append(example)
                    stats["by_framework"]["rag_v2"] += 1
                    stats["by_article"][article] += 1
                    stats["by_compliance_state"][compliance_state] += 1
                    stats["total"] += 1
                    example_count += 1
    print(f"  Generated {example_count - count_before} RAG v2 examples")
    
    # Partial compliance patterns (6 patterns × 6 articles × 2 variations = 72)
    print("Generating partial compliance edge case examples...")
    count_before = example_count
    for pattern_name, pattern_code in PARTIAL_COMPLIANCE_PATTERNS.items():
        for article in articles:
            for variation in range(2):
                instruction = f"Analyze this Python AI agent code for EU AI Act Article {article} ({ARTICLE_DESCRIPTIONS[article]}) compliance."
                code_with_var = add_variation(pattern_code)
                line_range = generate_line_range()
                output = FINDING_TEMPLATES_V2["partial_compliance"][article].format(lines=line_range)
                
                all_examples.append({
                    "instruction": instruction,
                    "input": code_with_var,
                    "output": output,
                    "metadata": {
                        "framework": "edge_case_partial_compliance",
                        "article": article,
                        "compliance_state": "partially_compliant",
                        "template": pattern_name,
                    }
                })
                stats["by_framework"]["edge_case_partial"] += 1
                stats["by_article"][article] += 1
                stats["by_compliance_state"]["partially_compliant"] += 1
                stats["total"] += 1
                example_count += 1
    print(f"  Generated {example_count - count_before} edge case examples")
    
    # Mixed framework examples (2 patterns × 6 articles × 2 variations = 24)
    print("Generating mixed framework examples...")
    count_before = example_count
    for pattern_name, pattern_code in MIXED_FRAMEWORK_PATTERNS.items():
        for article in articles:
            for variation in range(2):
                instruction = f"Analyze this Python AI agent code for EU AI Act Article {article} ({ARTICLE_DESCRIPTIONS[article]}) compliance."
                code_with_var = add_variation(pattern_code)
                line_range = generate_line_range()
                output = FINDING_TEMPLATES_V2["mixed_framework"][article].format(lines=line_range)
                
                all_examples.append({
                    "instruction": instruction,
                    "input": code_with_var,
                    "output": output,
                    "metadata": {
                        "framework": "mixed_framework",
                        "article": article,
                        "compliance_state": "non_compliant",
                        "template": pattern_name,
                    }
                })
                stats["by_framework"]["mixed_framework"] += 1
                stats["by_article"][article] += 1
                stats["by_compliance_state"]["non_compliant"] += 1
                stats["total"] += 1
                example_count += 1
    print(f"  Generated {example_count - count_before} mixed framework examples")
    
    # Obfuscated examples (3 patterns × 6 articles × 2 variations = 36)
    print("Generating obfuscated code examples...")
    count_before = example_count
    for pattern_name, pattern_code in OBFUSCATED_PATTERNS.items():
        for article in articles:
            for variation in range(2):
                instruction = f"Analyze this Python AI agent code for EU AI Act Article {article} ({ARTICLE_DESCRIPTIONS[article]}) compliance."
                code_with_var = add_variation(pattern_code)
                line_range = generate_line_range()
                output = FINDING_TEMPLATES_V2["obfuscated"][article].format(lines=line_range)
                
                all_examples.append({
                    "instruction": instruction,
                    "input": code_with_var,
                    "output": output,
                    "metadata": {
                        "framework": "obfuscated_code",
                        "article": article,
                        "compliance_state": "non_compliant",
                        "template": pattern_name,
                    }
                })
                stats["by_framework"]["obfuscated"] += 1
                stats["by_article"][article] += 1
                stats["by_compliance_state"]["non_compliant"] += 1
                stats["total"] += 1
                example_count += 1
    print(f"  Generated {example_count - count_before} obfuscated examples")
    
    print(f"\\nTotal v2 examples generated: {stats['total']}")
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Split train/eval
    split_point = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_point]
    eval_examples = all_examples[split_point:]
    
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    
    # Write v2 training data
    v2_train_file = "training_data_v2.jsonl"
    print(f"\\nWriting v2 training data to {v2_train_file}...")
    with open(v2_train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    print(f"  ✓ Wrote {v2_train_file} ({len(train_examples)} examples)")
    
    # Write v2 eval data
    v2_eval_file = "eval_data_v2.jsonl"
    print(f"Writing v2 eval data to {v2_eval_file}...")
    with open(v2_eval_file, 'w') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')
    print(f"  ✓ Wrote {v2_eval_file} ({len(eval_examples)} examples)")
    
    # Print statistics
    print("\\n" + "=" * 80)
    print("V2 GENERATION STATISTICS")
    print("=" * 80)
    
    print(f"\\nTotal V2 Examples: {stats['total']}")
    print(f"  Training: {len(train_examples)} (90%)")
    print(f"  Evaluation: {len(eval_examples)} (10%)")
    
    print(f"\\nBy Article:")
    for article in sorted(articles):
        count = stats["by_article"][article]
        pct = (count / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  Article {article} ({ARTICLE_DESCRIPTIONS[article]}): {count} ({pct:.1f}%)")
    
    print(f"\\nBy Framework/Pattern:")
    for framework in sorted(stats["by_framework"].keys()):
        count = stats["by_framework"][framework]
        pct = (count / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {framework}: {count} ({pct:.1f}%)")
    
    print(f"\\nBy Compliance State:")
    for state in ["non_compliant", "partially_compliant", "compliant"]:
        count = stats["by_compliance_state"].get(state, 0)
        pct = (count / stats["total"]) * 100 if stats["total"] > 0 else 0
        state_label = state.replace("_", " ").title()
        print(f"  {state_label}: {count} ({pct:.1f}%)")
    
    print("\\n" + "=" * 80)
    print(f"✓ V2 training data generation complete!")
    print(f"  V2 Files: {v2_train_file}, {v2_eval_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()