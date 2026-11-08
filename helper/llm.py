import re

import ollama
from pydantic import ValidationError

from model.ollama import AtomicStatements, PrecisionOutput, RecallOutput
from helper.log import log_llm_request
from helper.prompt import (
    atomic_statements_prompt,
    context_recall_prompt,
    precision_prompt,
)

_INVALID_CONTEXT_JUDGMENTS_ERROR = (
    "The evaluator returned incomplete or invalid context judgments: expected"
)
_INVALID_STATEMENT_JUDGMENTS_ERROR = (
    "The evaluator returned incomplete or invalid statement judgments: expected"
)

def llm(prompt, output_schema, retry_invalid_json=False):
    retry_prompt = prompt
    attempts = 2 if retry_invalid_json else 1

    for attempt in range(attempts):
        chat_arguments = {
            "model": 'hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:F16',
            "messages": [{
                "role": "user",
                "content": retry_prompt,
            }],
            "format": output_schema.model_json_schema(),
        }
        if attempt:
            chat_arguments["options"] = {"temperature": 0}

        response = ollama.chat(
            **chat_arguments,
        )
        result = response.message.content

        try:
            return output_schema.model_validate_json(result), result
        except ValidationError as error:
            if attempt + 1 < attempts:
                retry_prompt = (
                    "Your previous answer failed schema validation. "
                    "Redo the original task using the required output schema. "
                    "Pay special attention to the validation error and ensure "
                    "every item includes every required field with the "
                    "correct type. Return only a complete JSON response.\n\n"
                    f"Original task:\n{prompt}\n\n"
                    f"Previous invalid response:\n{result}\n\n"
                    f"Validation error:\n{error}"
                )
                continue

            log_llm_request("error", {
                "prompt": prompt,
                "response.message.content": result,
                "output_schema": output_schema.__name__,
                "error": str(error),
            })
            raise

    raise RuntimeError("LLM validation retry completed without a result.")


def calculate_context_precision(question, reference_answer, contexts, question_index):
    if not contexts:
        return 0.0
    
    print("\n--------------------------------\n")
    print(f"Number of context: {len(contexts)}")
    
    context_text = "\n\n".join([
            f"Context [index={i}]: {context}"
            for i, context in enumerate(contexts)
    ])
  
    prompt = precision_prompt(question, reference_answer, context_text)
    result, response_content = llm(prompt, PrecisionOutput)
    log_record = {
        "question_index": question_index,
        "question": question,
        "prompt": prompt,
        "response.message.content": response_content,
        "reference_answer": reference_answer,
        "context": contexts,
        "context_judgement": result.root,
    }

    judgments = _index_context_judgments(result.root, len(contexts))
    log_llm_request("success", log_record)
    return judgments


def calculate_context_recall(reference_answer, contexts, question_index):
    if not contexts:
        return 0.0

    extraction_prompt = atomic_statements_prompt(reference_answer)
    statement_result, extraction_response = llm(
        extraction_prompt,
        AtomicStatements,
    )
    
    statements = statement_result.statements
    extraction_record = {
        "question_index": question_index,
        "prompt": extraction_prompt,
        "response.message.content": extraction_response,
        "reference_answer": reference_answer,
        "statements": statements,
    }
    
    log_llm_request("success", extraction_record)

    if not statements:
        return 0.0

    prompt = context_recall_prompt(statements, contexts)
    result, response_content = llm(
        prompt,
        RecallOutput,
        retry_invalid_json=True,
    )
    
    judgments_by_index = {}
    invalid_keys = []
    for key, judgment in result.root.items():
        if not re.fullmatch(r"(0|[1-9]\d*)", key):
            invalid_keys.append(key)
        else:
            judgments_by_index[int(key)] = judgment.attributed

    expected_indices = set(range(len(statements)))
    if invalid_keys or judgments_by_index.keys() != expected_indices:
        error = (
            f"{_INVALID_STATEMENT_JUDGMENTS_ERROR} {len(statements)} "
            f"numeric indices from 0, got keys {list(result.root)}."
        )
        log_llm_request("error", {
            "question_index": question_index,
            "prompt": prompt,
            "response.message.content": response_content,
            "reference_answer": reference_answer,
            "statements": statements,
            "context": contexts,
            "error": error,
        })
        raise ValueError(error)

    log_llm_request("success", {
        "question_index": question_index,
        "prompt": prompt,
        "response.message.content": response_content,
        "reference_answer": reference_answer,
        "statements": statements,
        "context": contexts,
        "statement_judgement": {
            index: judgment.model_dump()
            for index, judgment in result.root.items()
        },
    })
    return sum(judgments_by_index.values()) / len(statements)


def _index_context_judgments(judgments, context_count):
    indexed_judgments = {}
    malformed_keys = []
    for key, verdict in judgments.items():
        match = re.fullmatch(r"Context \[index=(\d+)\]", key)
        if match is None:
            malformed_keys.append(key)
        else:
            index = int(match.group(1))
            indexed_judgments[index] = verdict

    expected_indices = set(range(context_count))

    if (
        malformed_keys
        or len(indexed_judgments) != len(judgments)
        or indexed_judgments.keys() != expected_indices
    ):
        raise ValueError(
            f"{_INVALID_CONTEXT_JUDGMENTS_ERROR} {context_count} "
            f"context keys indexed from 0, got {list(judgments)}."
        )
    return indexed_judgments