import streamlit as st
import time
import random
import requests

st.set_page_config(page_title="Agent Interface", layout="wide")

# Initialize session state for messages and agent internal state
if "messages" not in st.session_state:
    st.session_state.messages = []

default_agent_state = {
    "current_state": "Idle",
    "active_node": "None",
    "tool_calls": "None",
    "retrieval": "None",
    "memory": "None",
    "validation": "None",
    "generation": "None"
}
if "agent_state" not in st.session_state:
    st.session_state.agent_state = default_agent_state.copy()

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

# --- Layout ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("💬 Chat Window")
    chat_container = st.container(height=600)
    
    with chat_container:
        # Render existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

with col2:
    st.header("🧠 Agent Activity Panel")
    st.markdown("Monitor internal agent processes and states in real-time.")
    
    # Placeholders for real-time monitoring of specific fields
    metrics_container = st.container()
    with metrics_container:
        p_state = st.empty()
        p_node = st.empty()
        p_tools = st.empty()
        p_retrieval = st.empty()
        p_memory = st.empty()
        p_validation = st.empty()
        p_generation = st.empty()
        
    st.subheader("Activity Log")
    log_container = st.container(height=300)

def render_agent_state(state_dict):
    """Updates the metrics panel with current state."""
    p_state.info(f"**Current agent state:** {state_dict['current_state']}")
    p_node.info(f"**Active node in LangGraph:** {state_dict['active_node']}")
    p_tools.info(f"**Tool calls being executed:** {state_dict['tool_calls']}")
    p_retrieval.info(f"**Retrieval status:** {state_dict['retrieval']}")
    p_memory.info(f"**Memory updates:** {state_dict['memory']}")
    p_validation.info(f"**Validation results:** {state_dict['validation']}")
    p_generation.info(f"**Final response generation:** {state_dict['generation']}")

def add_log(msg):
    st.session_state.activity_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

def render_logs():
    with log_container:
        # Keep latest logs at bottom, scrollable via container height
        for line in st.session_state.activity_log[-20:]:  # show recent to not overflow easily
            st.text(line)

# Render initial state
render_agent_state(st.session_state.agent_state)
render_logs()

# --- Chat Input & Mock Processing ---
if prompt := st.chat_input("Type your message here..."):
    # Clear log for a new turn to keep it fresh
    st.session_state.activity_log = []
    
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Helper to mock streaming and state changes
            def transition(key, val, log_msg, t=0.5):
                st.session_state.agent_state[key] = val
                add_log(log_msg)
                render_agent_state(st.session_state.agent_state)
                render_logs()
                time.sleep(t)

            # Start of mock workflow
            st.session_state.agent_state["current_state"] = "Processing Input"
            add_log("Agent woken up. Starting thought process.")
            render_agent_state(st.session_state.agent_state)
            render_logs()
            time.sleep(0.5)
            
            # Node 1: Intent Parsing
            transition("active_node", "IntentRouter", "Entering IntentRouter node", 0.6)
            transition("current_state", "Determine Intent", "Extracting intent from user prompt", 0.8)
            transition("tool_calls", f"classifier.predict()", "Executing classifier tool", 1.0)
            
            # Node 2: Retrieval phase
            transition("active_node", "RetrieverNode", "Routing to Retrieval operations", 0.4)
            transition("current_state", "Searching Vectors", "Building query embedding", 0.7)
            transition("retrieval", "Searching FAISS index top_k=3", "Executing vector search", 1.2)
            transition("retrieval", "Found 3 docs (0.91, 0.85, 0.82 similarity)", "Search complete", 0.5)
            
            # Node 3: Update Memory
            transition("active_node", "MemoryManager", "Logging interaction to memory database", 0.4)
            transition("memory", f"Stored turn {len(st.session_state.messages)} context", "Memory commit successful", 0.6)
            
            # Node 4: Validation
            transition("active_node", "OutputValidator", "Checking constraints for planned output", 0.4)
            transition("validation", "Self-reflection: Passed", "Checked context adherence & safety", 1.0)
            
            # Simulate some state transitions purely for visual continuity in the UX 
            # while the backend processes the actual orchestrations natively.
            transition("active_node", "SupervisorAgent", "Routing request to backend API", 0.3)
            transition("current_state", "Network Request", "Calling POST /chat", 0.5)

            full_response = ""
            try:
                # Actual REST Request to the local FastAPI Backend
                headers = {"Authorization": "Bearer token_admin"} # Passing hardcoded mock token
                resp = requests.post("http://127.0.0.1:8000/chat", json={"message": prompt}, headers=headers)
                
                if resp.status_code == 200:
                    data = resp.json()
                    backend_reply = data.get("response", "")
                    fetched = data.get("context_fetched", 0)
                    
                    transition("retrieval", f"Aggregated {fetched} contexts", f"Backend dynamically fetched {fetched} items", 0.4)
                    transition("generation", "In Progress", "Retrieving final API answer", 0.2)
                    
                    # Stream actual result back to the frontend
                    for chunk in backend_reply.split(" "):
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(random.uniform(0.01, 0.05))
                        
                else:
                    full_response = f"Error from backend API: {resp.status_code} \n\n {resp.text}"
                    
            except Exception as e:
                full_response = f"Network Error. Ensure FastAPI backend is running! Details: {str(e)}"
            
            message_placeholder.markdown(full_response)
            transition("generation", "Completed", "Stream finalized successfully", 0.2)
            
            # Reset to idle
            st.session_state.agent_state = default_agent_state.copy()
            st.session_state.agent_state["memory"] = f"{len(st.session_state.messages)+1} turns active"
            render_agent_state(st.session_state.agent_state)
            add_log("Agent returned to Idle state.")
            render_logs()
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # We don't necessarily have to rerun here because we've updated everything in containers manually,
    # but a rerun will clean up the ephemeral placeholders if you wanted a more permanent feeling page.
