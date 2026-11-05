import ollama

from model.ollama import PrecisionOutput
from helper.prompt import precision_prompt

def llm(prompt, output_schema):
    response = ollama.chat(
        model='hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q8_0',
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format=output_schema.model_json_schema(),
        options={
            "temperature": 0,
            "num_predict": 4096
        }
    )
    result = response.message.content
    return output_schema.model_validate_json(result)

def calculate_context_precision(question, reference_answer, contexts):
    if not contexts:
        return 0.0
    print(len(contexts))
    context_text = "\n\n".join([
            f"Context [index={i}]: {context}"
            for i, context in enumerate(contexts)
    ])
  
    prompt = precision_prompt(question, reference_answer, context_text)
    result = llm(prompt, PrecisionOutput)
    return _index_context_judgments(result.judgments, "verdict", len(contexts))

def _index_context_judgments(items, value_attribute, context_count):
    indices = [item.context_index for item in items]
    expected_indices = set(range(context_count))
    returned_indices = set(indices)

    if len(indices) == context_count and returned_indices == expected_indices:
        index_offset = 0
    elif (len(indices) == context_count and returned_indices == set(range(1, context_count + 1))):
        index_offset = 1
    else:
        raise ValueError(
            "The evaluator returned incomplete or invalid context judgments: "
            f"expected {context_count} unique indices in "
            f"{sorted(expected_indices)}, got {indices}."
        )

    return {
        item.context_index - index_offset: getattr(item, value_attribute)
        for item in items
    }