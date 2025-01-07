import os

# Set environment variable to avoid KMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
import gc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_model_and_tokenizer(model_name, device):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    print("Loading model with 4-bit optimization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        quantization_config=bnb_config
    ).to(device)

    return tokenizer, model

def chunk_text(text, tokenizer, max_input_length):
    """
    Splits the input text into chunks that fit within the model's token limit.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]

    # Split input_ids into manageable chunks
    chunks = [
        input_ids[i : i + max_input_length]
        for i in range(0, len(input_ids), max_input_length)
    ]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarizer(text, tokenizer, model, device, max_input_length=1024, max_new_tokens=150):
    """
    Summarizes a text that might be too long by chunking it.
    """
    print("Chunking text...")
    chunks = chunk_text(text, tokenizer, max_input_length)

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=30,
                length_penalty=2.0,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chunk_summaries.append(summary)

    # Combine all chunk summaries into one and summarize again
    combined_summary = " ".join(chunk_summaries)
    print("Generating final summary from combined chunks...")
    final_inputs = tokenizer(combined_summary, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

    with torch.no_grad():
        final_outputs = model.generate(
            **final_inputs,
            max_new_tokens=max_new_tokens,
            min_length=30,
            length_penalty=2.0,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
    final_summary = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return final_summary

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def main():
    input_file = "text.txt"
    output_file = "summary.txt"

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        return

    print("Reading input file...")
    text = read_file(input_file)

    print("Loading model and tokenizer...")
    model_name = "prithivMLmods/Llama-Chat-Summary-3.2-3B"
    tokenizer, model = load_model_and_tokenizer(model_name, device)

    print("Summarizing text...")
    max_input_length = 1024  # Adjust to your model's token limit
    max_new_tokens = 150  # Length of the generated summary
    summary = summarizer(text, tokenizer, model, device, max_input_length, max_new_tokens)

    print("Writing summary to output file...")
    write_file(output_file, summary)

    print("Summary written to", output_file)
    gc.collect()

if __name__ == "__main__":
    main()