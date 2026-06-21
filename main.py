
"""
main.py

Developer Test Console

Production deployment uses FastAPI.

This file exists only for:

- Retrieval testing
- Routing testing
- Local debugging
- Evaluation

Run:

python main.py --mode hybrid
"""

import argparse


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEFAULT_COLLECTION = (
    "research_papers"
)

DEFAULT_CHUNK_DIR = (
    "data/chunks"
)


# --------------------------------------------------
# CHAT
# --------------------------------------------------

def run_chat():

    from generate import (
        generate_answer
    )

    chat_history = []

    print("\nFirstRAG Chat")
    print("Type 'exit' to quit.")

    while True:

        query = input(
            "\nYou: "
        )

        if (
            query
            .lower()
            .strip()
            == "exit"
        ):
            break

        response = (
            generate_answer(

                query=query,

                collection_name=
                DEFAULT_COLLECTION,

                chunk_dir=
                DEFAULT_CHUNK_DIR,

                chat_history=
                chat_history,

                k=3,
            )
        )

        answer = (
            response["answer"]
        )

        print(
            f"\nAssistant:\n"
            f"{answer}"
        )

        chat_history.append({

            "role":
            "user",

            "content":
            query,
        })

        chat_history.append({

            "role":
            "assistant",

            "content":
            answer,
        })


# --------------------------------------------------
# DENSE TEST
# --------------------------------------------------

def run_dense():

    from retrieve import (
        retrieve
    )

    query = input(
        "\nQuery: "
    )

    response = (
        retrieve(

            query=query,

            collection_name=
            DEFAULT_COLLECTION,

            k=5,
        )
    )

    print(
        "\nMetrics:"
    )

    print(
        response["metrics"]
    )

    for rank, result in enumerate(
        response["results"],
        start=1,
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: "
            f"{result.score:.4f}"
        )

        print(
            result.payload.get(
                "paper_name"
            )
        )


# --------------------------------------------------
# BM25 TEST
# --------------------------------------------------

def run_bm25():

    from retrieve_bm25 import (
        bm25_retrieve
    )

    query = input(
        "\nQuery: "
    )

    response = (
        bm25_retrieve(

            query=query,

            chunk_dir=
            DEFAULT_CHUNK_DIR,

            k=5,
        )
    )

    print(
        response["metrics"]
    )

    for rank, result in enumerate(
        response["results"],
        start=1,
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            result["score"]
        )


# --------------------------------------------------
# HYBRID TEST
# --------------------------------------------------

def run_hybrid():

    from retrieve_reranked_hybrid import (
        retrieve_hybrid
    )

    query = input(
        "\nQuery: "
    )

    response = (
        retrieve_hybrid(

            query=query,

            collection_name=
            DEFAULT_COLLECTION,

            chunk_dir=
            DEFAULT_CHUNK_DIR,

            k=5,
        )
    )

    print(
        "\nMetrics:"
    )

    print(
        response["metrics"]
    )

    for rank, (
        parent_id,
        parent_doc,
        score,
    ) in enumerate(

        response["results"],

        start=1,
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            parent_doc
            .metadata
            .get(
                "paper_name"
            )
        )

        print(
            f"Score:"
            f"{score:.4f}"
        )


# --------------------------------------------------
# ROUTER TEST
# --------------------------------------------------

def run_router():

    from router import (
        decide_route
    )

    query = input(
        "\nQuery: "
    )

    result = (
        decide_route(
            query=query,
            history="",
        )
    )

    print(
        "\nRouting Result:"
    )

    print(result)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def run_eval():

    from evaluate_hybrid import (
        main
    )

    main()


def run_ragas():

    from evaluate_own_ragas_v2 import (
        main
    )

    main()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = (
        argparse.ArgumentParser(
            description=
            "FirstRAG Developer Console"
        )
    )

    parser.add_argument(

        "--mode",

        required=True,

        choices=[

            "chat",

            "dense",

            "bm25",

            "hybrid",

            "router",

            "eval",

            "ragas",
        ],
    )

    args = (
        parser.parse_args()
    )

    mode = args.mode

    if mode == "chat":
        run_chat()

    elif mode == "dense":
        run_dense()

    elif mode == "bm25":
        run_bm25()

    elif mode == "hybrid":
        run_hybrid()

    elif mode == "router":
        run_router()

    elif mode == "eval":
        run_eval()

    elif mode == "ragas":
        run_ragas()


if __name__ == "__main__":
    main()

