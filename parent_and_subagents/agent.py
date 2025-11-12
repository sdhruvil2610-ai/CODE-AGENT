
import os
import logging
import google.cloud.logging
from dotenv import load_dotenv
from .callback_logging import log_query_to_model, log_model_response
from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import exit_loop
from google.genai import types

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()
model_name = os.getenv("MODEL", "gemini-2.5-pro")
print(model_name)

def append_to_state(tool_context: ToolContext, field: str, response: str) -> dict[str, str]:
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}
def get_latest_state(tool_context: ToolContext) -> dict[str, str]:
    """
    Extracts the latest code draft and feedback from the shared state.
    Ensures each iteration works on the most recent version only.
    """
    drafts = tool_context.state.get("CODE_DRAFT", [])
    feedbacks = tool_context.state.get("CRITICAL_FEEDBACK", [])

    latest_draft = drafts[-1] if drafts else ""
    latest_feedback = feedbacks[-1] if feedbacks else ""

    logging.info("[State Sync] Using latest CODE_DRAFT and CRITICAL_FEEDBACK.")
    return {
        "latest_code_draft": latest_draft,
        "latest_feedback": latest_feedback
    }

def handoff_to(tool_context: ToolContext, agent_name: str) -> dict[str, str]:
    """
    Explicitly transfers control to the named sub-agent inside the current parent.
    Used to enforce deterministic next-step routing (e.g., writer → critic).
    """
    logging.info(f"[handoff_to] Transferring to agent: {agent_name}")
    return {"transfer_to_agent": {"agent_name": agent_name}}

code_critic = Agent(
    name="code_critic",
    model=model_name,
    description="Reviews the latest code draft and provides actionable feedback for improvement.",
    instruction="""
CODE_DRAFT:
{ latest_code_draft? }

INSTRUCTIONS:
- Review the latest draft from CODE_DRAFT.
- Validate:
  1) Syntax and imports (compiles cleanly)
  2) ADK structure (Agents, tools, Loop/Sequential flow)
  3) Runnable standalone script (env setup, logging)
  4) Clear docstrings / no extraneous prose
  5) No duplicate code blocks

BEHAVIOR RULES:
- If you are mistakely called first sent user prompt as feedback to writer agent using Append CRITICAL_FEEDBACK
- If improvement is needed:
  * Append CRITICAL_FEEDBACK with specific, concise actions.
  * Then call handoff_to("code_writer").
  When Maximam Iterations reached exit the loop
- Never respond to the user or other peers.
""",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[get_latest_state, append_to_state, handoff_to],
    disallow_transfer_to_parent=True,   # stay within loop, no fallback to greeter
)

code_writer = Agent(
    name="code_writer",
    model=model_name,
    description="Writes or refines Python ADK code using the most recent critique feedback.",
    instruction="""
PROMPT:
{ PROMPT? }

CODE_DRAFT:
{ latest_code_draft? }

CRITICAL_FEEDBACK:
{ latest_feedback? }

INSTRUCTIONS:
-Never transfer control to parent agent, Always transfer control to code_critic agent using handoff tool this is critical do not forgot that.
- You are the **Code Writer** in a refinement loop.
- Always read and apply the most recent feedback from the code_critic agent stored in CRITICAL_FEEDBACK before writing new code.
- If no feedback exists, generate the first version from PROMPT.
- If feedback exists, revise ONLY the most recent draft based on it.
- Append the new code to CODE_DRAFT  using append_to_state.
- Then call handoff_to("code_critic") to continue the loop.
- Always call handoff_to("code_critic") to continue the loop when necessary so that loop execute sucessfully we do not wated to transfer control to rool agent once you receive prompt from it remember it
-Never transfer control to parent agent, Always transfer control to code_critic agent using handoff tool this is critical do not forgot that.
- Never produce prose or explanations, only pure code.
- Do not exit or call peers other than code_critic.
- Never call peers other than code_critic, and never speak to the user.
- Always include necessary imports, environment setup, and comments if needed inside the code.
-Only use the tools that we have described not any other tool remeber that

When you describe your idea, I’ll interpret it using the following ADK framework guidelines:

1. **Agent Types:**
   - **Root Agent:** Handles user input and orchestrates other agents.
   - **SequentialAgent:** Runs sub-agents in a defined order (pipeline-style).
   - **LoopAgent:** Iterates between agents (for iterative improvements or feedback loops).
   - **Tool Agent:** Wraps utility functions like reading/writing files, appending state, etc.

2. **Core Imports:**
   - `from google.adk import Agent`
   - `from google.adk.agents import SequentialAgent, LoopAgent`
   - `from google.adk.tools import exit_loop`
   - `from google.adk.tools.tool_context import ToolContext`
   - `from google.genai import types`
   - `from dotenv import load_dotenv`
   - `import os, logging, google.cloud.logging`

3. **Environment Variables:**
   - MODEL → Which Gemini model to use (e.g. gemini-2.5-pro)
   - GOOGLE_API_KEY → Required API key
   - Additional optional config (like USE_VERTEX or TEMPERATURE)

4. **Coding Conventions:**
   - Always define `append_to_state()` for tool-based state persistence.
   - Use structured prompts with sections like PROMPT, CODE_DRAFT, CRITICAL_FEEDBACK and CRITICAL_FEEDBACK.
   - Always log queries and responses with `before_model_callback` and `after_model_callback`.
   - Never use async; keep all code synchronous like the Loop Agent reference.

5. **Output Requirements:**
   - The final output must be Python code only, enclosed in fenced code blocks (```python ... ```).
   - No prose explanations — only runnable code.
   - Use consistent indentation and explicit imports.
- Behavior rules:
  1. If PROMPT exists and no CODE_DRAFT exists → generate initial implementation.
  2. If CRITICAL_FEEDBACK exists → revise the latest CODE_DRAFT accordingly.
  3. Append your complete code to 'CODE_DRAFT' using append_to_state.
  4. Then **transfer control** to the `code_critic` agent for review.
  5. Do NOT end the loop or exit yourself — let code_critic decide when to stop.
  6.After writing the draft, return control to the loop controller (do NOT call peers, do NOT talk to the user).
BEHAVIOR RULES:
1️⃣ If first iteration → generate initial code.
2️⃣ If CRITICAL_FEEDBACK exists → revise last draft.
3️⃣ Append final code to CODE_DRAFT.
4️⃣ Call handoff_to("code_critic").
5️⃣ Never call exit_loop or interact with the user.

OUTPUT FORMAT:
Output raw Python code only (no markdown fences).
""",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[get_latest_state, append_to_state, handoff_to],
    disallow_transfer_to_parent=True,   # keeps control within dev_loop
)



dev_loop = LoopAgent(
    name="dev_loop",
    description="Starts always from code_writer and Iterates between code_writer and code_critic.",
    sub_agents=[code_writer, code_critic],
    max_iterations=5, 
)
final_presenter = Agent(
    name="final_presenter",
    model=model_name,
    description="Outputs final improved code as text.",
    instruction="""
CODE_DRAFT:
{ latest_code_draft? }

INSTRUCTIONS:
- Consolidate and refine the final code draft.
- Output ONLY the Python code formatted as:
\\`\\`\\`python
# final code
\\`\\`\\`
No explanations or commentary.
""",
    generate_content_config=types.GenerateContentConfig(temperature=0),
    disallow_transfer_to_parent=True,
disallow_transfer_to_peers=True,

)
builder_team = SequentialAgent(
    name="builder_team",
    description="Runs the dev loop and final presenter.",
    sub_agents=[dev_loop, final_presenter],
    

)

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="Greets the user and starts the code generator process.",
    instruction="""
Welcome! I’ll help you generate **Python code for a new ADK Agent system.**

Please describe the agent you want to build — include its purpose, input/output behavior, 
and any specific logic you want (like using Gemini, file tools, or custom callbacks).

Transfer the findings by using tools to builder team

Once you describe your agent idea, I’ll translate it into a complete ADK-compatible implementation using these standards.
""",
    tools=[get_latest_state, append_to_state],
    sub_agents=[builder_team],
    disallow_transfer_to_parent=True,
disallow_transfer_to_peers=True,

)

