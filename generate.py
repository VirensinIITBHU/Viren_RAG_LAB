"""
generate.py

Production Generation Service

Features
--------
✓ Source Grounded Answers
✓ Citation Aware
✓ Hybrid Retrieval Support
✓ Follow-Up Handling
✓ Query Rewriting
✓ Llama 3.3 70B
✓ Frontend Configurable Retrieval
✓ Hallucination Guard
✓ FastAPI Ready
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from router import decide_route
from retrieve_reranked_hybrid import (
    retrieve_hybrid,
    format_sources,
)

load_dotenv()

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = (
    "meta-llama/llama-3.3-70b-instruct"
)

# ==================================================
# OPENROUTER CLIENT
# ==================================================

_client = None


def get_client():

    global _client

    if _client is None:

        _client = OpenAI(
            api_key=os.getenv(
                "OPEN_ROUTER_KEY"
            ),
            base_url=
            "https://openrouter.ai/api/v1",
        )

    return _client


# ==================================================
# HISTORY HELPERS
# ==================================================

def format_history(
    chat_history,
):

    if not chat_history:

        return (
            "No previous conversation."
        )

    return "\n".join(

        f"{msg['role']}: "
        f"{msg['content']}"

        for msg in chat_history[-8:]
    )


def format_user_history(
    chat_history,
):

    return "\n".join(

        msg["content"]

        for msg in chat_history

        if msg["role"] == "user"
    )


# ==================================================
# QUERY REWRITE
# ==================================================

def rewrite_query(
    query: str,
    chat_history: list,
):

    if not chat_history:
        return query

    ambiguous_words = {

        "it",
        "this",
        "that",
        "they",
        "them",
        "their",
        "its",
        "those",
        "these",
    }

    if not any(

        word in query.lower().split()

        for word in ambiguous_words

    ):
        return query

    history = (
        format_user_history(
            chat_history
        )
    )

    prompt = f"""
Rewrite the user's latest query
into a standalone question.

Conversation History:
{history}

Latest Query:
{query}

Return only the rewritten query.
"""

    try:

        response = (
            get_client()
            .chat
            .completions
            .create(

                model=MODEL_NAME,

                temperature=0,

                messages=[
                    {
                        "role":
                        "user",

                        "content":
                        prompt,
                    }
                ],
            )
        )

        return (
            response
            .choices[0]
            .message.content
            .strip()
        )

    except Exception:

        return query

# ==================================================
# CONTEXT BUILDER
# ==================================================

def build_context(
    retrieval_results,
):

    context_blocks = []
    source_mapping = []

    for idx, (
        chunk_id,
        payload,
        score,
    ) in enumerate(
        retrieval_results,
        start=1,
    ):

        source_id = f"SOURCE_{idx}"
        paper_name = payload.get("paper_name", "Unknown")
        page = payload.get("page", "?")
        text = payload.get("text", "")

        block = f"""
{source_id}

Paper:
{paper_name}

Chunk:
{chunk_id}

Page:
{page}

Text:
{text}
"""

        context_blocks.append(block)

        # THE FIX: Added "content": text so the frontend can display the preview!
        source_mapping.append({
            "source_id": source_id,
            "paper_name": paper_name,
            "chunk_id": chunk_id,
            "page": page,
            "score": round(score, 4),
            "content": text  
        })

    return (
        "\n\n-----------------------------\n\n".join(context_blocks),
        source_mapping,
    )


# ==================================================
# CHAT RESPONSE
# ==================================================

def generate_chat_response(
    query: str,
):

    response = (

        get_client()
        .chat
        .completions
        .create(

            model=MODEL_NAME,

            temperature=0.3,

            messages=[
                {
                    "role":
                    "user",

                    "content":
                    query,
                }
            ],
        )
    )

    return (
        response
        .choices[0]
        .message.content
    )


# ==================================================
# RAG GENERATION
# ==================================================

def generate_rag_response(
    query: str,
    history: str,
    retrieval_results,
):

    context, source_mapping = (
        build_context(
            retrieval_results
        )
    )

    prompt = f"""
You are a research assistant.

Use ONLY the provided sources.

Rules:

1. Every factual claim MUST cite:
   [SOURCE_X]

2. Use multiple citations when needed.

3. If information is not present
   in the sources, say:

   "I could not find this information
   in the uploaded documents."

4. Do not use outside knowledge.

Conversation:
{history}

Sources:
{context}

Question:
{query}

Answer:
"""

    response = (

        get_client()
        .chat
        .completions
        .create(

            model=MODEL_NAME,

            temperature=0,

            messages=[
                {
                    "role":
                    "user",

                    "content":
                    prompt,
                }
            ],
        )
    )

    answer = (
        response
        .choices[0]
        .message.content
    )

    return (
        answer,
        source_mapping,
    )


# ==================================================
# MAIN ENTRYPOINT
# ==================================================

def generate_answer(

    query: str,

    collection_name: str,

    chat_history: list = None,

    dense_k: int = 20,

    bm25_k: int = 20,

    rerank_candidates: int = 20,

    final_k: int = 5,
):

    if chat_history is None:

        chat_history = []

    history = (
        format_history(
            chat_history
        )
    )

    route_result = (
        decide_route(
            query=query,
            history=history,
        )
    )

    route = (
        route_result[
            "route"
        ]
    )

    # ==========================================
    # CHAT
    # ==========================================

    if route == "CHAT":

        answer = (
            generate_chat_response(
                query
            )
        )

        return {

            "answer":
            answer,

            "route":
            route,

            "sources":
            [],

            "retrieval_metrics":
            {},

            "routing_metrics":
            route_result,
        }

    # ==========================================
    # FOLLOWUP
    # ==========================================

    standalone_query = (
        rewrite_query(
            query,
            chat_history,
        )
    )

    # ==========================================
    # RETRIEVE
    # ==========================================

    retrieval_response = (
        retrieve_hybrid(

            query=
            standalone_query,

            collection_name=
            collection_name,

            dense_k=
            dense_k,

            bm25_k=
            bm25_k,

            rerank_candidates=
            rerank_candidates,

            final_k=
            final_k,
        )
    )

    retrieval_results = (
        retrieval_response[
            "results"
        ]
    )

    retrieval_metrics = (
        retrieval_response[
            "metrics"
        ]
    )
    print("\nRetrieved Sources:\n")

    for chunk_id, payload, score in retrieval_results:

        print(
            payload.get("paper_name"),
            score,
        )

    # ==========================================
    # GENERATE
    # ==========================================

    answer, sources = (
        generate_rag_response(

            query=query,

            history=history,

            retrieval_results=
            retrieval_results,
        )
    )

    return {

        "answer":
        answer,

        "route":
        route,

        "sources":
        sources,

        "retrieval_metrics":
        retrieval_metrics,

        "routing_metrics":
        route_result,
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    response = generate_answer(

        query=
        "What is Attention Is All You Need?",

        collection_name=
        "research_papers",

        chat_history=[],

        dense_k=20,

        bm25_k=20,

        rerank_candidates=20,

        final_k=5,
    )

    print()

    print(
        response["answer"]
    )

    print()

    print(
        response["sources"]
    )

    print()

    print(
        response["retrieval_metrics"]
    )