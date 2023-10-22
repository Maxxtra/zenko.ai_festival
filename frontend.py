# %load_ext tensorboard
# %tensorboard --logdir results/runs
from transformers import AutoModelForCausalLM, AutoTokenizer

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Define the path to the trained model
new_model_path = "llama-2-7b-miniguanaco"

# Load the trained model
loaded_model = AutoModelForCausalLM.from_pretrained(new_model_path)

# Load the same tokenizer used for training
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Hi, I am the Director of Customer Relations. How may I help you?")
question = input()

while(question!='QUIT'):
	
	# Tokenize the input data
	input_ids = tokenizer.encode(question, return_tensors="pt")

	# Generate text with the model
	output = loaded_model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True)

	# Decode the generated output
	generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

	print("Answer:", generated_text)
	print("Ask me another question, or type QUIT if you want to exit the conversation")
	question = input()
