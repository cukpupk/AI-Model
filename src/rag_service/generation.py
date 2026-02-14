from litellm import completion

from rag_service.config import get_settings


SYSTEM_PROMPT = """You are an enterprise assistant.
Use only the provided context to answer.
If information is missing, explicitly say you do not have enough context.
Always be concise and factual.
"""


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"source: {chunk['source']}\n"
            f"text: {chunk['text']}\n"
        )
    return "\n".join(blocks)


def generate_answer(query: str, chunks: list[dict]) -> str:
    settings = get_settings()
    context = build_context(chunks)
    user_prompt = (
        "Answer the question using the context below.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Provide a direct answer and mention uncertainty if needed."
    )

    try:
        completion_kwargs = {
            "model": settings.litellm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.litellm_temperature,
            "max_tokens": settings.litellm_max_tokens,
        }
        if settings.litellm_api_base:
            completion_kwargs["api_base"] = settings.litellm_api_base

        response = completion(
            **completion_kwargs,
        )
        return response.choices[0].message.content or "No answer generated."
    except Exception as exc:  # pragma: no cover
        preview = chunks[0]["text"][:280] if chunks else ""
        return (
            "LLM generation is not configured (missing/invalid provider API key). "
            "Set credentials in .env (e.g., OPENAI_API_KEY) and retry. "
            f"Retrieved context preview: {preview}\n"
            f"Technical detail: {type(exc).__name__}"
        )
