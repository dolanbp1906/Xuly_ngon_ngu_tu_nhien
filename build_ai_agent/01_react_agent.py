"""Section 5 — ReAct agent: reason, call a tool, observe, answer.

Mirrors the guide's LangGraph example:
    tools -> model -> create_react_agent -> invoke
"""

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from llm_factory import make_llm, visible_text

QUESTION = "What is 123 * 456?"
EXPECTED = 123 * 456


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def search_web(query: str) -> str:
    """Simulated search tool."""
    return f"Results for {query}..."


def main() -> None:
    tools = [multiply, search_web]
    llm = make_llm()
    agent = create_react_agent(
        llm,
        tools,
        prompt=(
            "You are a ReAct agent. For any product of two integers, call the "
            "multiply tool. Do not calculate the product yourself. For factual "
            "lookup, call search_web. After the tool result, answer in one sentence."
        ),
    )

    response = agent.invoke(
        {"messages": [("user", QUESTION)]},
        config={"recursion_limit": 8},
    )

    print(f"question: {QUESTION}")
    saw_multiply = False
    for message in response["messages"]:
        role = message.__class__.__name__
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                print(f"action: {call['name']}({call['args']})")
                if call["name"] == "multiply":
                    saw_multiply = True
        content = visible_text(getattr(message, "content", ""))
        if role == "ToolMessage" and content:
            print(f"observation: {content}")
        elif role == "AIMessage" and content and not tool_calls:
            print(f"answer: {content}")

    answer = visible_text(response["messages"][-1].content)
    passed = saw_multiply and str(EXPECTED) in answer
    print(f"expected: {EXPECTED}")
    print(f"pass@1: {'pass' if passed else 'fail'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
