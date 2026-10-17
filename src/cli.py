import os
import sys
import textwrap
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from retrieve import search

# Prompt template: concise, instruction-focused
SYSTEM_PROMPT = (
    "You are an HR assistant. Answer strictly based on the provided context.\n"
    "-If user greets you, greet them back in friendly manner.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Be concise and clear.\n"
)

USER_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


def format_context(results) -> str:
    """Format search results into a readable context string for prompting.

    Args:
        results: Iterable of tuples ``(score: float, chunk: str, md: dict)``
            from the retriever. ``md`` is expected to include ``doc_id`` and
            ``chunk_id`` keys.

    Returns:
        A single string where each chunk is preceded by a header containing
        source document info and similarity score, with blocks separated by
        a blank line.
    """
    ctx_blocks = []
    for score, chunk, md in results:
        header = f"[source: {md['doc_id']} | chunk: {md['chunk_id']} | score: {score:.3f}]"
        ctx_blocks.append(f"{header}\n{chunk}")
    return "\n\n".join(ctx_blocks)


def generate_answer(question: str, context: str) -> str:
    """Generate a concise answer from the Gemini model using retrieved context.

    Loads the API key from the environment (via ``.env``), builds a
    constrained prompt with system guidance and the provided context, and
    returns the model's text output.

    Args:
        question: The user's question to answer.
        context: The formatted context string produced by ``format_context``.

    Returns:
        The model response text. If the response has no direct ``text`` field,
        attempts to assemble text from candidates. Falls back to
        "(No response)" if nothing is available.

    Raises:
        RuntimeError: If ``GEMINI_API_KEY`` is not set in the environment/.env.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing in environment/.env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash") # gemini-2.0-flash 

    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(question=question, context=context)
    resp = model.generate_content(prompt)
    # Handle candidates safely
    text = getattr(resp, "text", None)
    if not text and hasattr(resp, "candidates") and resp.candidates:
        parts = []
        for c in resp.candidates:
            if c.content and c.content.parts:
                parts.extend([getattr(p, "text", "") for p in c.content.parts])
        text = "\n".join([t for t in parts if t])
    return text or "(No response)"


def main():
    """Entry point for the RAG-based HR chatbot CLI.

    - One-shot mode: pass a question as CLI arguments to get a single answer.
    - Interactive mode: prompts repeatedly until the user enters 'exit'/'quit',
      sends EOF, or interrupts with the keyboard.
    """
    # Chat-like CLI: interactive loop; also supports one-shot if args provided
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

    print("HR RAG Chatbot\nType your question below. Type 'exit' or 'quit' to end.\n")
    while True:
        try:
            user_inp = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_inp:
            continue
        if user_inp.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        results = search(user_inp, k=5)
        context = format_context(results)
        answer = generate_answer(user_inp, context)
        print("Bot:\n" + answer + "\n")


if __name__ == "__main__":
    main()