"""
router.py

Production Query Router

Pipeline
--------
User Query
    ↓
Rule Router
    ↓

CHAT
RETRIEVE
FOLLOWUP
AMBIGUOUS

    ↓

LLM Router (fallback only)

Features
--------
✓ Rule-first routing
✓ Follow-up detection
✓ Confidence scoring
✓ Latency profiling
✓ OpenRouter support
✓ FastAPI ready
✓ Cheap (LLM only when needed)
"""

import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
# ROUTING RULES
# ==================================================

CHAT_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good evening",
    "how are you",
    "who are you",
}

FOLLOWUP_WORDS = {
    "it",
    "this",
    "that",
    "they",
    "them",
    "their",
    "those",
    "these",
}

RETRIEVAL_TERMS = {

    # AI / RAG

    "rag",
    "lora",
    "bert",
    "gpt",
    "llm",
    "embedding",
    "embeddings",
    "vector",
    "vectors",
    "transformer",
    "attention",
    "retrieval",
    "reranker",
    "chunking",
    "langchain",
    "qdrant",
    "chroma",

    # Research

    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "document",
    "documents",

    # Medical

    "diabetes",
    "metformin",
    "cancer",
    "glucose",
    "insulin",
}


# ==================================================
# HELPERS
# ==================================================

def tokenize(text: str):

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def build_response(
    route: str,
    source: str,
    confidence: float,
    start_time: float,
):

    return {
        "route": route,
        "source": source,
        "confidence": round(
            confidence,
            2,
        ),
        "latency_ms": round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2,
        ),
    }


# ==================================================
# RULE ROUTER
# ==================================================

def rule_route(
    query: str,
):

    start_time = (
        time.perf_counter()
    )

    query = (
        query.strip()
    )

    query_lower = (
        query.lower()
    )

    tokens = (
        tokenize(
            query_lower
        )
    )

    # ------------------------------------------
    # CHAT
    # ------------------------------------------

    if query_lower in CHAT_PATTERNS:

        return build_response(
            route="CHAT",
            source="RULE",
            confidence=1.0,
            start_time=start_time,
        )

    # ------------------------------------------
    # FOLLOW-UP
    # ------------------------------------------

    if len(tokens) <= 6:

        followup_hits = sum(

            token
            in FOLLOWUP_WORDS

            for token
            in tokens
        )

        if followup_hits > 0:

            return build_response(
                route="FOLLOWUP",
                source="RULE",
                confidence=0.80,
                start_time=start_time,
            )

    # ------------------------------------------
    # RETRIEVAL
    # ------------------------------------------

    retrieval_hits = sum(

        token
        in RETRIEVAL_TERMS

        for token
        in tokens
    )

    if retrieval_hits > 0:

        confidence = min(
            retrieval_hits / 2,
            1.0,
        )

        return build_response(
            route="RETRIEVE",
            source="RULE",
            confidence=confidence,
            start_time=start_time,
        )

    # ------------------------------------------
    # QUESTIONS
    # ------------------------------------------

    if (
        query.endswith("?")
        or len(tokens) > 3
    ):

        return build_response(
            route="AMBIGUOUS",
            source="RULE",
            confidence=0.50,
            start_time=start_time,
        )

    return build_response(
        route="CHAT",
        source="RULE",
        confidence=0.60,
        start_time=start_time,
    )


# ==================================================
# LLM ROUTER
# ==================================================

def llm_route(
    query: str,
    history: str = "",
):

    start_time = (
        time.perf_counter()
    )

    client = (
        get_client()
    )

    prompt = f"""
You are a query routing classifier.

Return ONLY one word.

CHAT
RETRIEVE
FOLLOWUP

Definitions:

CHAT
- greetings
- small talk
- conversational requests

RETRIEVE
- factual questions
- document questions
- research questions
- technical explanations

FOLLOWUP
- depends on previous context

Conversation:
{history}

User:
{query}
"""

    response = (

        client.chat
        .completions
        .create(

            model=
            "openai/gpt-4o-mini",

            temperature=0,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    )

    decision = (

        response
        .choices[0]
        .message.content
        .strip()
        .upper()
    )

    if decision not in {
        "CHAT",
        "RETRIEVE",
        "FOLLOWUP",
    }:

        decision = (
            "RETRIEVE"
        )

    return build_response(
        route=decision,
        source="LLM",
        confidence=0.90,
        start_time=start_time,
    )


# ==================================================
# PUBLIC API
# ==================================================

def decide_route(
    query: str,
    history: str = "",
):

    rule_result = (
        rule_route(
            query
        )
    )

    if rule_result["route"] in {
        "CHAT",
        "RETRIEVE",
        "FOLLOWUP",
    }:

        return rule_result

    return llm_route(
        query=query,
        history=history,
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_queries = [

        "hi",

        "what is lora",

        "how does attention work",

        "tell me more",

        "what about it",

        "metformin mechanism",

        "summarize this paper",

        "explain rag fusion",
    ]

    for query in test_queries:

        result = (
            decide_route(
                query
            )
        )

        print(
            f"\nQuery: {query}"
        )

        print(
            result
        )