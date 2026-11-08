import math

from helper.llm import (
    calculate_context_recall,
    calculate_context_precision,
)

def evaluate(data):
    results = []
    documents = zip(data["question"], data["candidate"], data["reference"])
    for question_index, (question, candidate, reference) in enumerate(documents):
        judgments = (
            calculate_context_precision(question, reference, candidate, question_index)
            if candidate else {}
        )

        binary_relevance = [judgments[index] for index in range(len(judgments))]

        result = {
            "question": question,
            "candidate": candidate,
            "reference": reference,
            "context_precision": calculate_average_precision(binary_relevance),
            "context_recall": calculate_context_recall(reference, candidate, question_index),
            "ndcg@5": calculate_ndcg(binary_relevance, k=5),
        }
        results.append(result)
    return results


def _build_evaluation_data(dataset, retrieve_contexts):
    evaluation_data = {
        "question": [],
        "candidate": [],
        "reference": [],
    }

    for row in dataset:
        question = str(row["question"]).lower()

        evaluation_data["question"].append(question)
        evaluation_data["reference"].append(str(row["context"]).lower())
        evaluation_data["candidate"].append([
            str(context).lower()
            for context in retrieve_contexts(question)
        ])
    return evaluation_data


def evaluate_dense_retriever(retriever, dataset):
    def retrieve_contexts(question):
        return retriever.search(str(question).lower())

    print("\n\nDense retriever evaluation:")
    
    data = _build_evaluation_data(dataset, retrieve_contexts)
    return evaluate(data)


def calculate_average_precision(judgments):
    relevant_count = sum(judgments)

    if relevant_count == 0:
        return 0.0

    cumulative_relevant = 0
    precision_sum = 0.0

    for rank, relevance in enumerate(judgments, start=1):
        if relevance == 1:
            cumulative_relevant += 1
            precision_at_rank = cumulative_relevant / rank
            precision_sum += precision_at_rank
    return precision_sum / relevant_count


def calculate_ndcg(judgments, k=5):
    top_judgments = judgments[:k]
    if not top_judgments:
        return 0.0

    dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(top_judgments, start=1)
    )
    
    ideal_dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(
            sorted(judgments, reverse=True)[:k],
            start=1,
        )
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0