import os
import sys

from groq import Groq
from dotenv import load_dotenv

from retrieve import search


# Load environment variables from .env
load_dotenv()


# System instructions for the HR chatbot
SYSTEM_PROMPT = (
    "You are an HR assistant. Answer strictly based on the provided context.\n"
    "- If the user greets you, greet them back in a friendly manner.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Never invent HR policies or company information.\n"
    "- Be concise, clear, and professional.\n"
)

USER_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


def format_context(results) -> str:
    """
    Format retrieved document chunks into context for the LLM.

    Args:
        results: Iterable of tuples (score, chunk, metadata).

    Returns:
        Formatted context string.
    """
    ctx_blocks = []

    for score, chunk, md in results:
        header = (
            f"[source: {md.get('doc_id', 'unknown')} | "
            f"chunk: {md.get('chunk_id', -1)} | "
            f"score: {score:.3f}]"
        )

        ctx_blocks.append(f"{header}\n{chunk}")

    return "\n\n".join(ctx_blocks)


def generate_answer(question: str, context: str) -> str:
    """
    Generate an HR answer using a Groq-hosted language model.

    Args:
        question: User's question.
        context: Relevant HR document context.

    Returns:
        Generated answer.

    Raises:
        RuntimeError: If GROQ_API_KEY is missing.
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY missing in environment/.env"
        )

    client = Groq(api_key=api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return answer.strip() if answer else "(No response)"


def main():
    """
    Run the HR RAG chatbot in one-shot or interactive CLI mode.
    """

    # One-shot mode:
    # python -m src.cli What is the leave policy?
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:]).strip()

        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            return

        results = search(question, k=5)
        context = format_context(results)

        answer = generate_answer(question, context)

        print("\nBot:\n" + answer)
        return

    # Interactive mode
    print(
        "HR RAG Chatbot using Groq\n"
        "Type your question below. Type 'exit' or 'quit' to end.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            results = search(user_input, k=5)
            context = format_context(results)

            answer = generate_answer(user_input, context)

            print("Bot:\n" + answer + "\n")

        except Exception as error:
            print(f"Error: {error}\n")


if __name__ == "__main__":
    main()