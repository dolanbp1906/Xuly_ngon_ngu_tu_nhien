"""Section 6 — manual tool calling with a validation gate.

The model only proposes a tool call. This script checks the arguments, then
either runs the tool or refuses it, and sends that observation back.
"""

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llm_factory import make_llm, visible_text

QUESTION = "What is 123 * 456?"
EXPECTED = 123 * 456
MAX_ABS = 1_000_000


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def search_web(query: str) -> str:
    """Simulated search tool."""
    return f"Results for {query}..."


TOOLS = {"multiply": multiply, "search_web": search_web}


def validate(name: str, args: dict) -> str | None:
    """Return a refusal reason, or None when the call is safe to run."""
    if name not in TOOLS:
        return f"unknown tool {name}"
    if name != "multiply":
        return None
    for key in ("a", "b"):
        value = args.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{key} must be an integer"
        if abs(value) > MAX_ABS:
            return f"{key}={value} is outside the safe range +/-{MAX_ABS}"
    return None


def execute(name: str, args: dict) -> str:
    reason = validate(name, args)
    if reason:
        print(f"validate: reject {name}({args}) - {reason}")
        return f"rejected: {reason}"
    print(f"validate: accept {name}({args})")
    result = TOOLS[name].invoke(args)
    print(f"execute: {result}")
    return str(result)


def run_turn(llm, messages: list, question: str) -> str:
    print(f"\nquestion: {question}")
    ai = llm.invoke(messages + [HumanMessage(content=question)])
    if not ai.tool_calls:
        answer = visible_text(ai.content)
        print(f"answer (no tool): {answer}")
        return answer

    messages = [*messages, HumanMessage(content=question), ai]
    for call in ai.tool_calls:
        observation = execute(call["name"], call["args"])
        messages.append(
            ToolMessage(content=observation, tool_call_id=call["id"])
        )
    final = llm.invoke(messages)
    answer = visible_text(final.content)
    print(f"answer: {answer}")
    return answer


def demo_rejected_call() -> None:
    """A call the model did not make: blocked before multiply runs."""
    print("\nquestion: synthetic unsafe call")
    fake = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "multiply",
                "args": {"a": 10, "b": 10**12},
                "id": "unsafe-demo",
                "type": "tool_call",
            }
        ],
    )
    observation = execute(fake.tool_calls[0]["name"], fake.tool_calls[0]["args"])
    print(f"observation: {observation}")
    if not observation.startswith("rejected:"):
        raise SystemExit("unsafe call was not blocked")


def main() -> None:
    llm = make_llm().bind_tools([multiply, search_web])
    system = SystemMessage(
        content=(
            "Call the multiply tool for products of two integers. "
            "Do not calculate the product yourself. "
            "After the tool result, answer in one sentence."
        )
    )
    answer = run_turn(llm, [system], QUESTION)
    if str(EXPECTED) not in answer:
        raise SystemExit(f"expected {EXPECTED} in the final answer")
    print(f"expected: {EXPECTED}")
    print("pass@1: pass")
    demo_rejected_call()


if __name__ == "__main__":
    main()
