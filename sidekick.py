from typing import Annotated, List, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import uuid
import asyncio
from datetime import datetime

# Import your updated tools
from sidekick_tools import (
    get_prateek_cv_info,
    get_github_info,
    get_contact_info,
    get_project_details,
    get_work_experience_details,
    answer_as_prateek,
    get_prateek_context_for_evaluator,
)

from sidekick_tools import playwright_tools, other_tools

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        # New system message per requirements
        system_message = f"""You are a friendly, slightly humorous assistant whose sole purpose is to help anyone learn about *Prateek Ghorawat*. 
Never discuss anything unrelated to Prateek. Never speak ill of anyone or anything. Keep the conversation engaging: sprinkle in jokes or light humor about Prateek’s hobbies, achievements, or AI adventures, but always be professional.

The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Success criteria:
{state["success_criteria"]}

Reply either by asking a clarifying question about Prateek’s profile or by giving your final answer about Prateek. 
If you need more information about his CV, GitHub, projects, or background, ask clearly.

Example question:
Question: Could you clarify which project on Prateek’s GitHub you’d like details on?
"""

        if state.get("feedback_on_work"):
            system_message += f"""
Previous feedback said your last answer missed something. Feedback: {state['feedback_on_work']}
Please adjust your response about Prateek accordingly or ask any needed question."""

        # Insert/update system message in state
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_message)] + messages
        else:
            for m in messages:
                if isinstance(m, SystemMessage):
                    m.content = system_message

        # Invoke
        response = self.worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def worker_router(self, state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "evaluator"

    def format_conversation(self, messages: List[Any]) -> str:
        conv = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                conv += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                conv += f"Assistant: {m.content or '[tool usage]'}\n"
        return conv

    def evaluator(self, state: State) -> State:
        last_resp = state["messages"][-1].content
        system = """You are an evaluator assessing whether the assistant’s response about Prateek is:
1. Fully accurate about his CV, GitHub, projects, education, and experience.
2. Never off-topic or discussing anything unrelated.
3. Professional in tone yet occasionally humorous in the style requested.
4. Respectful—no negative remarks about Prateek or others.
5. Asking for clarification if needed.

Provide feedback, set success_criteria_met=True only if all above are satisfied, else False. 
Set user_input_needed=True if more info or clarification is required."""

        user_msg = f"""Conversation:
{self.format_conversation(state['messages'])}

Success criteria:
{state['success_criteria']}

Assistant’s last reply:
{last_resp}
"""
        if state["feedback_on_work"]:
            user_msg += f"Previous feedback: {state['feedback_on_work']}\n"

        eval_msgs = [SystemMessage(content=system), HumanMessage(content=user_msg)]
        result = self.evaluator_llm_with_output.invoke(eval_msgs)
        return {
            "messages": [{"role": "assistant", "content": f"Evaluator Feedback: {result.feedback}"}],
            "feedback_on_work": result.feedback,
            "success_criteria_met": result.success_criteria_met,
            "user_input_needed": result.user_input_needed,
        }

    def route_based_on_evaluation(self, state: State) -> str:
        return "END" if (state["success_criteria_met"] or state["user_input_needed"]) else "worker"

    async def build_graph(self):
        gb = StateGraph(State)
        gb.add_node("worker", self.worker)
        gb.add_node("tools", ToolNode(tools=self.tools))
        gb.add_node("evaluator", self.evaluator)
        gb.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        gb.add_edge("tools", "worker")
        gb.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        gb.add_edge(START, "worker")
        self.graph = gb.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        state = {
            "messages": message,
            "success_criteria": success_criteria or "Provide accurate, engaging info only about Prateek",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        result = await self.graph.ainvoke(state, config={"configurable": {"thread_id": self.sidekick_id}})
        user = {"role": "user", "content": message}
        # AIMessage objects—access .content
        assistant_msg = result["messages"][-2]
        reply = {"role": "assistant", "content": assistant_msg.content}
        feedback_msg = result["messages"][-1]
        feedback = {"role": "assistant", "content": feedback_msg.content}
        return history + [user, reply, feedback]


    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
