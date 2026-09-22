from typing import Annotated, Any, Dict, List, Literal, Sequence, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langsmith import traceable
from pydantic import BaseModel
from logger import logger
from security import SecurityGuardrails
from tools.knowledge_search import HybridRetriever
from tools.mcp_client import MockMCPServer
from tools.python_analysis import PythonDataAnalyzer
from langchain_openrouter import ChatOpenRouter

# 1. Define State Schema (Memory)
# This schema drives conversational memory and internal routing states.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: Literal["Supervisor Agent", "Retrieval Agent", "Research Agent", "Response Agent"]
    current_intent: str
    subtasks: List[str]
    retrieved_context: List[str]
    recursion_depth: int
    final_response: str


class RouteDecision(BaseModel):
    """Structured output schema for the supervisor's routing decision."""
    next_agent: Literal["Retrieval Agent", "Research Agent", "Response Agent"]
    current_intent: str

# 2. Node Definitions (Agents)
@traceable(run_type="chain", name="SupervisorAgent")
async def supervisor_agent(state: AgentState) -> Dict[str, Any]:
    """
    Responsible for: Intent understanding, Task decomposition, and Agent routing.
    """
    logger.info("supervisor_agent_started")

    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)

    routing_messages = [
        (
            "system",
             """You are a supervisor agent. You decide the user intent and does the agent routing. There are other agents named such as

                "Supervisor Agent", "Retrieval Agent", "Research Agent" and "Response Agent".

                Retrieval Agent
                    Responsible for:
                    ● RAG operations
                    ● Vector search
                Research Agent
                    Responsible for:
                    ● Deep investigation
                    ● Recursive exploration
                Response Agent
                    Responsible for:
                    ● Final answer generation

                Make sure to name only one agent for the next action.
                """,
        ),
        *state["messages"],
    ]

    # structured output
    structured_model = llm.with_structured_output(RouteDecision, method="json_schema")
    decision = await structured_model.ainvoke(routing_messages)

    logger.info("supervisor_agent_routing", next_agent=decision.next_agent)
    return {"next_agent": decision.next_agent, "current_intent": decision.current_intent}


@traceable(run_type="chain", name="RetrievalAgent")
async def retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """
    Responsible for: RAG operations & Vector search (Hybrid).
    """
    logger.info("retrieval_agent_started")
    context = state.get("retrieved_context", [])
    
    retriever = HybridRetriever(use_mock=True)
    
    # Normally we extract the query and embed it via an LLM.
    dummy_query = "payment issues"
    dummy_embedding = [0.1] * 1536  # Default dimension mock
    
    results = await retriever.search(
        query=dummy_query, 
        dense_embedding=dummy_embedding,
        namespace="default",
        metadata_filters={"department": "payments"}
    )
    
    # This loop is re-entered once per subtask (see research_agent), and since the
    # query is currently the same hardcoded string each time, dedupe against what's
    # already been collected so identical docs aren't appended repeatedly.
    existing = set(context)
    extracted_docs = [r["text"] for r in results if r["text"] not in existing]
    logger.info("retrieval_agent_completed", new_docs=len(extracted_docs))

    return {"retrieved_context": context + extracted_docs, "next_agent": "Research Agent"}


@traceable(run_type="chain", name="ResearchAgent_RLM")
async def research_agent(state: AgentState) -> Dict[str, Any]:
    """
    Responsible for: Deep investigation, Recursive Langauge Model (RLM).
        1. Explore document collections
        2. Generate Python-based search plans
        3. Decompose large tasks
        4. Retrieve targeted sections
        5. Call sub-agents recursively
        6. Aggregate results

    """
    depth = state.get("recursion_depth", 0)
    logger.info("research_agent_started", depth=depth)
    
    # Copy rather than alias: nodes must not mutate the incoming state's containers in place.
    context = list(state.get("retrieved_context", []))
    subtasks = list(state.get("subtasks", []))
    
    # 1. Decompose large tasks if starting fresh
    if depth == 0 and not subtasks:
        logger.info("research_agent_decomposing", action="creating_search_plan")
        # In a real app, an LLM would read the prompt and decompose here.
        # Mocking the decomposition of: "Summarize Q3 payment outages"
        subtasks = ["batch_search_incidents_q3", "batch_search_runbooks"]

    # 2 & 5. Call sub-agents recursively and Retrieve targeted sections
    if subtasks and depth < 3:
        current_task = subtasks.pop(0)
        logger.info("research_agent_recursive_dive", current_task=current_task, remaining=len(subtasks))
        
        # We might use MCP to check real time owners
        if "incident" in current_task:
            mcp = MockMCPServer()
            # 5. Call sub-agent (MCP abstraction)
            health = await mcp.get_service_health("pay-gateway")
            context.append(f"MCP Health check result: {health}")
            
        # Push back into State for the next node iteration (Recursion)
        return {
            "subtasks": subtasks, 
            "recursion_depth": depth + 1, 
            "retrieved_context": context,
            "next_agent": "Retrieval Agent" # Loop back out to find specific docs for the sub-task
        }
        
    # 3 & 4 & 6. Generate Python-based search plans & Aggregate results
    logger.info("research_agent_aggregating", action="python_code_generation")
    
    analyzer = PythonDataAnalyzer()
    
    # Generate python code (Simulated LLM plan)
    python_plan = """
def analyze_batches(contexts):
    # filter by topic
    outages = [c for c in contexts if 'outage' in c]
    # analyze batches separately & aggregate
    return summarize(outages)
"""
    
    aggregation_result = analyzer.analyze(data=context, analysis_code=python_plan)
    
    # Add final aggregated insights into context for Response Agent
    context.append(f"RLM Aggregation: {aggregation_result}")
    
    return {
        "recursion_depth": depth + 1,
        "retrieved_context": context,
        "next_agent": "Response Agent"
    }


@traceable(run_type="chain", name="ResponseAgent")
async def response_agent(state: AgentState) -> Dict[str, Any]:
    """
    Responsible for: Final answer generation enforcing Guardrails and Brand constraints.
    """
    logger.info("response_agent_started")
    context = state.get("retrieved_context", [])

    # Synthesize the final answer from accumulated context while guarding against prompt injections.
    if context:
        draft_response = "Here is what I found:\n" + "\n".join(f"- {item}" for item in context)
    else:
        draft_response = "I could not find relevant information for your request."

    final_checked = SecurityGuardrails.validate_agent_output(draft_response)
    
    return {"next_agent": END, "final_response": final_checked}

# 3. Build Graph
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Register all our nodes
    workflow.add_node("Supervisor Agent", supervisor_agent)
    workflow.add_node("Retrieval Agent", retrieval_agent)
    workflow.add_node("Research Agent", research_agent)
    workflow.add_node("Response Agent", response_agent)
    
    # 4. Wiring Graph Logic
    workflow.add_edge(START, "Supervisor Agent")
    
    # Explicit path maps so an unknown/mistyped next_agent value fails at compile time
    # rather than at first invocation.
    workflow.add_conditional_edges(
        "Supervisor Agent",
        lambda x: x["next_agent"]
    )
    workflow.add_conditional_edges(
        "Retrieval Agent",
        lambda x: x["next_agent"]
    )
    workflow.add_conditional_edges(
        "Research Agent",
        lambda x: x["next_agent"]
    )
    
    # Response Agent signals the end of the flow
    workflow.add_edge("Response Agent", END)
    
    # We compile the graph using an ephemeral memory saver or SQLite later to persist conversational memory
    return workflow.compile()

# Exportable ready-compiled graph
graph = build_agent_graph()
