
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists,
    k=60
):

    scores = defaultdict(float)

    documents = {}

    for results in result_lists:

        for rank, doc in enumerate(
            results,
            start=1
        ):

            doc_id = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content
            )

            scores[doc_id] += (
                1 / (k + rank)
            )

            documents[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids
    ]
