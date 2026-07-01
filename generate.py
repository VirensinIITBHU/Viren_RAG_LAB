"""
generate.py

Production Generation Service (Parent-Child Aware)

Features
--------
✓ Source Grounded Answers
✓ Citation Aware
✓ Hybrid Retrieval Support
✓ Follow-Up Handling
✓ Query Rewriting
✓ Llama 3.1 8B
✓ Frontend Configurable Retrieval
✓ Hallucination Guard
✓ FastAPI Ready
"""

import os
import time
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

MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

# ==================================================
# OPENROUTER CLIENT
# ==================================================

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPEN_ROUTER_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


# ==================================================
# HISTORY HELPERS
# ==================================================

def format_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    return "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in chat_history[-8:]
    )


def format_user_history(chat_history):
    return "\n".join(
        msg["content"]
        for msg in chat_history
        if msg["role"] == "user"
    )


# ==================================================
# QUERY REWRITE
# ==================================================

def rewrite_query(query: str, chat_history: list):
    if not chat_history:
        return query

    ambiguous_words = {
        "it", "this", "that", "they", "them",
        "their", "its", "those", "these",
    }

    if not any(word in query.lower().split() for word in ambiguous_words):
        return query

    history = format_user_history(chat_history)

    prompt = f"""
Rewrite the user's latest query into a standalone question.

Conversation History:
{history}

Latest Query:
{query}

Return only the rewritten query.
"""
    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query


# ==================================================
# CONTEXT BUILDER
# ==================================================

def build_context(retrieval_results):
    context_blocks = []
    source_mapping = []

    # FIX 1: Unpack parent_id instead of chunk_id
    for idx, (parent_id, payload, score) in enumerate(retrieval_results, start=1):
        source_id = f"SOURCE_{idx}"
        paper_name = payload.get("paper_name", "Unknown")
        child_chunk_id = payload.get("child_chunk_id", "?")
        
        raw_page = payload.get("page")
        if raw_page is None:
            raw_page = payload.get("page_number", "?")
            
        if isinstance(raw_page, int):
            page = str(raw_page + 1)
        else:
            page = str(raw_page)
            
        text = payload.get("text", "")

        block = f"""
{source_id}
Paper: {paper_name}
Parent ID: {parent_id}
Triggered by Child ID: {child_chunk_id}
Page: {page}

Text:
{text}
"""
        context_blocks.append(block.strip())

        source_mapping.append({
            "source_id": source_id,
            "paper_name": paper_name,
            "parent_id": parent_id,
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

def generate_chat_response(query: str):
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content


# ==================================================
# RAG GENERATION
# ==================================================

def generate_rag_response(query: str, history: str, retrieval_results):
    context, source_mapping = build_context(retrieval_results)

    prompt = f"""
You are a highly precise research assistant.

Use ONLY the provided sources to answer the question.

Strict Rules:
1. Every factual claim MUST be immediately followed by its citation, e.g., "The model achieved 94% accuracy [SOURCE_1]."
2. Use multiple citations when combining facts: "Fact A [SOURCE_1] and Fact B [SOURCE_2]."
3. If information is not present in the sources, you must explicitly state: "I could not find this information in the uploaded documents."
4. Do not use outside knowledge. Do not hallucinate data.

Conversation:
{history}

Sources:
{context}

Question:
{query}

Answer:
"""

    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    return answer, source_mapping


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
    request_start = time.perf_counter()

    if chat_history is None:
        chat_history = []

    history = format_history(chat_history)
    route_result = decide_route(query=query, history=history)
    route = route_result["route"]

    # ==========================================
    # CHAT ROUTE
    # ==========================================
    if route == "CHAT":
        answer = generate_chat_response(query)
        return {
            "answer": answer,
            "route": route,
            "sources": [],
            "retrieval": {},
            "retrieval_metrics": {},
            "observability": {
                "routing": {
                    "selected_route": route,
                    **route_result,
                },
                "retrieval": {},
                "rewrite_ms": 0,
                "generation_ms": 0,
                "total_request_ms": 0,
                "model": MODEL_NAME,
            },
            "routing_metrics": route_result,
        }

    # ==========================================
    # RAG ROUTE
    # ==========================================
    rewrite_start = time.perf_counter()
    standalone_query = rewrite_query(query, chat_history)
    rewrite_ms = round((time.perf_counter() - rewrite_start) * 1000, 2)

    retrieval_response = retrieve_hybrid(
        query=standalone_query,
        collection_name=collection_name,
        dense_k=dense_k,
        bm25_k=bm25_k,
        rerank_candidates=rerank_candidates,
        final_k=final_k,
    )

    retrieval_results = retrieval_response["results"]
    retrieval_metrics = retrieval_response["metrics"]
    
    print("\nRetrieved Sources:\n")
    # FIX 1 (Continued): Update loop variables
    for parent_id, payload, score in retrieval_results:
        print(f"{payload.get('paper_name')} | Score: {score:.4f}")

    generation_start = time.perf_counter()
    answer, sources = generate_rag_response(
        query=query,
        history=history,
        retrieval_results=retrieval_results,
    )
    generation_ms = round((time.perf_counter() - generation_start) * 1000, 2)
    total_request_ms = round((time.perf_counter() - request_start) * 1000, 2)

    return {
        "answer": answer,
        "route": route,
        "sources": sources,
        "retrieval": retrieval_metrics,
        "observability": {
            "routing": {
                "selected_route": route, 
                **route_result
            },
            "retrieval": retrieval_metrics,
            "rewrite_ms": rewrite_ms,
            "generation_ms": generation_ms,
            "total_request_ms": total_request_ms,
            "model": MODEL_NAME,
        },
        "routing_metrics": route_result,
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    response = generate_answer(
        query="What is Attention Is All You Need?",
        collection_name="research_papers",
        chat_history=[],
        dense_k=20,
        bm25_k=20,
        rerank_candidates=20,
        final_k=5,
    )

    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(response["answer"])

    print("\n" + "="*50)
    print("SOURCES:")
    print("="*50)
    for s in response["sources"]:
        print(f"[{s['source_id']}] {s['paper_name']} (Parent ID: {s['parent_id']})")

    print("\n" + "="*50)
    print("METRICS:")
    print("="*50)
    # FIX 2: Fixed the KeyError by calling the correct dictionary key
    print(response["retrieval"])