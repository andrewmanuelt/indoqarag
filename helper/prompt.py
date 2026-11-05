def precision_prompt(question, reference, context):
    prompt = f"""
    You are an information retrieval evaluator.

    Evaluate the relevance of each retrieved context
    based on the question and reference answer.

    Question:
    {question}

    Reference Answer:
    {reference}

    Retrieved Contexts:
    {context}

    Evaluation criteria:
    - verdict = 1: The context contains information relevant
    to answering the question and supports the reference answer.
    - verdict = 0: The context is irrelevant or does not support
    the reference answer.

    Rules:
    1. Evaluate every context independently.
    2. Do not use external knowledge.
    3. Return context_index exactly as shown in each "Context [index]" label.
    4. Indices start at 0.
    5. Provide a concise reason for each judgment.
    6. Return one judgment for every retrieved context.
    7. Follow the required JSON schema exactly.
    """
    
    return prompt