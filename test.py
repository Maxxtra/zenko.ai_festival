from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "Maxtra/llama-2-7b-frestival"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input text
input_text = "This is a test sentence. "

# Encode the input
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
