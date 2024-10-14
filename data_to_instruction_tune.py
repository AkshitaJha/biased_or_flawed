import pandas as pd
import json
import random

# Load the dataset
with open('train.json', 'r') as file:
    squad_data = json.load(file)

# List of instructions with the condition to answer with "Not in background"
instructions = [
    "Answer the question using the context provided. If the answer is not present, respond with 'Not in background.'",
    "Based on the given context, answer the question. If insufficient information is available, reply 'Not in background.'",
    "Respond to the following question using the provided context. If the answer isn't found, state 'Not in background.'",
    "Use the background information to answer the question. If the context doesn't contain the answer, write 'Not in background.'",
    "Determine the answer using the provided text. If the answer cannot be found, say 'Not in background.'",
    "Provide an answer based on the context. If the context lacks sufficient information, respond with 'Not in background.'",
    "Using the context above, answer the question. If no answer is available, indicate 'Not in background.'",
    "Answer the question using the given information. If the answer isn't included, respond with 'Not in background.'",
    "Utilize the context to answer the question. If the answer is not present, reply with 'Not in background.'",
    "Find the answer within the provided context. If the context does not include the answer, state 'Not in background.'",
    "Answer the question using the context provided. ",
    "Based on the given context, answer the question.",
    "Respond to the following question using the provided context.",
    "Use the background information to answer the question.",
    "Determine the answer using the provided text.",
    "Provide an answer based on the context.",
    "Using the context above, answer the question.",
    "Answer the question using the given information.",
    "Utilize the context to answer the question.",
    "Find the answer within the provided context. ",
]

# Prepare a list to hold the instruction-tuned data
instruction_tuned_data = []

# Loop through each topic in the SQuAD dataset
for topic in squad_data['data']:
    for paragraph in topic['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            qa_type = 'disambig'
            question = qa['question']
            if qa.get('is_impossible', False):  # Check if the question is impossible to answer
                answer_text = "Not in background."
                qa_type = "ambig"
            elif qa['answers']:  # Check if there are any answers
                answer_text = qa['answers'][0]['text']  # Using the first answer
            else:
                answer_text = "Not in background."

            # Select a random instruction
            instruction = random.choice(instructions)

            # Create an instruction example
            instruction_example = {
                "ques_type": qa_type,
                "instruction": instruction,
                "input": f"Context: {context}\nQuestion: {question}",
                "response": answer_text,
            }
            instruction_tuned_data.append(instruction_example)

# Convert to a DataFrame or save directly to JSON
df_instruction_tuned = pd.DataFrame(instruction_tuned_data)

# Save to CSV file
df_instruction_tuned.to_csv('instr_train.csv', index=False)

# Display the first few rows of the instruction-tuned data
print(df_instruction_tuned.head())
