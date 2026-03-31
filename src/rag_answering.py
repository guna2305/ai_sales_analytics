from typing import List, Tuple
from src.rag_pipeline import Doc


def build_context_block(results: List[Tuple[float, Doc]]) -> str:
    lines = []
    for i, (score, doc) in enumerate(results, start=1):
        lines.append(f"[Source {i} | score={score:.3f}] {doc.text}")
    return "\n".join(lines)


def template_answer(query: str, results: List[Tuple[float, Doc]]) -> str:
    if not results:
        return (
            "I do not have enough retrieved context to answer this question accurately. "
            "Please rebuild the knowledge base or ask a question related to uploaded sales data and forecast outputs."
        )

    top_context = "\n".join([f"- {doc.text}" for _, doc in results[:3]])

    return (
        f"Answer based only on retrieved project data:\n\n"
        f"Question: {query}\n\n"
        f"Relevant findings:\n{top_context}\n\n"
        f"Grounded summary:\n"
        f"The answer above is derived only from the retrieved sales analytics and forecast knowledge base."
    )


def build_grounded_prompt(query: str, results: List[Tuple[float, Doc]]) -> str:
    context = build_context_block(results)

    prompt = f"""
You are an AI sales analytics assistant.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT invent values, stores, categories, or trends.
3. If the context is insufficient, clearly say so.
4. Give a business-style answer in short paragraphs or bullets.
5. Mention source numbers where useful.

User question:
{query}

Context:
{context}

Now answer the question in a grounded and professional way.
"""
    return prompt.strip()