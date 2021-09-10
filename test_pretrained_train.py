#!/usr/bin/env python3


'''
This is a simple script that uses the huggingfaces library to finetune a GPT2 model 
on some text.
'''

# Imports
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import argparse

# Get the arguments
parser = argparse.ArgumentParser(description='Finetune GPT2 on some text')
parser.add_argument('--text', type=str, help='The text to finetune on')
parser.add_argument('--model_name', type=str, default='gpt2', help='The name of the model to finetune')
parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for finetuning')
parser.add_argument('--epochs', type=int, default=3, help='The number of epochs to finetune for')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='The learning rate to use for finetuning')
args = parser.parse_args()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

# Load the model
config = GPT2Config.from_pretrained(args.model_name)
model = GPT2LMHeadModel.from_pretrained(args.model_name, config=config)
model = model.to(device)

# Read the text from file args.text.
with open(args.text, 'r') as f:
    text = f.read()

# Encode the text
text = tokenizer.encode(text, return_tensors='pt').to(device)


# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
model.train()
for epoch in range(args.epochs):
    optimizer.zero_grad()
    model_output = model(text, labels=text)
    loss = model_output[0]
    logits = model_output[1]
    loss.backward()
    optimizer.step()

    # Decode the logits to text.
    text_logits = logits[:, :-1]
    predicted_text = torch.argmax(text_logits, dim=-1)
    predicted_text = tokenizer.decode(predicted_text[0])
    print("predicted_text:", predicted_text)
   

    # Print training statistics.
    print(f'Epoch: {epoch}\tLoss: {loss}')

# Save the model.
# model.save_pretrained('./')

# Use the trained language model to make predictions.
# Load the tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

# Load the model.
config = GPT2Config.from_pretrained(args.model_name)
# model = GPT2LMHeadModel.from_pretrained(args.model_name, config=config)
model = model.to(device)

# Encode the prompt text.
# prompt_text = 'The president'
prompt_text = input('Enter prompt: ')
prompt_text = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

# Add a batch dimension.
prompt_text = prompt_text.unsqueeze(0)

# Generate the predictions.
model.eval()
print("prompt_text:", prompt_text.shape)
# Remove first dimension from prompt_text.
prompt_text = prompt_text.squeeze(0)
print("prompt_text:", prompt_text.shape)
predictions = model.generate(prompt_text, max_length=400)

# Decode  the predictions.
predictions = tokenizer.decode(predictions[0])

# Print the prompt and the decoded prediction.
print(f'Prompt: {prompt_text}')
print(f'Decoded prediction: {predictions}')

