from typing import List, Tuple
from src.rag_pipeline import Doc

def build_context_block(results: List[Tuple[float, Doc]]) -> str:
    lines = []
    for score, doc in results:
        lines.append(f"- (score={score:.3f}) {doc.text}")
    return "\n".join(lines)

def template_answer(query: str, context: str) -> str:
    # Cloud-safe fallback: still “AI-like” but deterministic and grounded in retrieved context
    return (
        "Answer (grounded on retrieved project data):\n\n"
        f"Question: {query}\n\n"
        "Relevant context:\n"
        f"{context}\n\n"
        "Summary:\n"
        "Based on the retrieved context above, the response should focus on the listed KPIs/forecast values/trends.\n"
    )