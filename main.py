#!/usr/bin/env python3


'''
This is a script that uses the replies from one language models to train another language model to give better replies.
'''

import argparse
import os
import sys
import logging

from transformers import pipeline

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import argparse

# Initialize visdom.
import visdom
vis = visdom.Visdom()

OUTPUT_FILE = 'completed.txt'

PROMPT = \
'''Chat protocol of support hotline.\n''' \
'''Support: How can I help you?\n''' \
'''Customer's question:'''                                                                          

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainGPT2:
    def __init__(self):
        # Load the tokenizer
        model_name = 'pytorch_model.bin'
        self.epochs = 1
        self.load_model('gpt2')
        self.smoothed_loss = 0

    def load_model(self, model_name='gpt2'):
        # Load the model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, config=self.config)
        self.model = model.to(device)

    def train(self, text, iteration):
        # Encode the text
        text = self.tokenizer.encode(text, return_tensors='pt').to(device)


        # Train the model
        model = self.model
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            model_output = model(text, labels=text)
            loss = model_output[0]
            logits = model_output[1]
            loss.backward()
            optimizer.step()

            # Decode the logits to text.
            text_logits = logits[:, :-1]
            predicted_text = torch.argmax(text_logits, dim=-1)
            predicted_text = self.tokenizer.decode(predicted_text[0])
            if True:
                print("predicted_text:", predicted_text)
                # Print training statistics.
                print(f'Epoch: {epoch}\tLoss: {loss}')

            # Log the loss to Visdom and set a title for the plot.
            vis.line(X=torch.tensor([iteration]), Y=torch.tensor([loss]), win='loss', update='append', opts=dict(title='loss'))

            # Smoothed loss for logging.
            self.smoothed_loss = 0.99 * self.smoothed_loss + 0.01 * loss
            print(f'{iteration}\tLoss: {self.smoothed_loss}')

            # Log the smoothed_loss using Visdom and set a title for the plot.
            vis.line(X=torch.tensor([iteration]), Y=torch.tensor([self.smoothed_loss]), win='smoothed loss', update='append', opts=dict(title='smoothed loss'))

        # Save the model.
    def complete(self, text):
        prompt_text = self.tokenizer.encode(text, return_tensors='pt').to(device)

        # Add a batch dimension.
        prompt_text = prompt_text.unsqueeze(0)

        # Generate the predictions.
        self.model.eval()

        # Remove first dimension from prompt_text.
        prompt_text = prompt_text.squeeze(0)
        predictions = self.model.generate(prompt_text, max_length=50)

        # Decode  the predictions.
        predictions = self.tokenizer.decode(predictions[0])

        return predictions


class StandardGPT2:
    def __init__(self):
        pipeline_name = "text-generation"
        # pipeline_kwargs = {'temperature': 1.0}                         
        pipeline_kwargs = {}                         
        self.pipeline_class = pipeline(pipeline_name, **pipeline_kwargs)                                         

    def complete(self, input_str):
        completions = self.pipeline_class(input_str)
        completions = [completion['generated_text'] for completion in completions]

        with open(OUTPUT_FILE, 'w') as f:                                                               
            for completion in completions:                                                              
                f.write(completion + '\n')                                                              

        return completions

    def get_customer_message(self):
        customer_message = self.complete(PROMPT)
        return customer_message

def main():
    standard_gpt2 = StandardGPT2()
    train_gpt2 = TrainGPT2()        
    i = 0
    smoothed_quality_int = 1
    while True:
        customer_message = standard_gpt2.get_customer_message()[0]
        next_prompt = customer_message + "\nSupport's high quality reply:"
        support_message = train_gpt2.complete(next_prompt)
        next_prompt = support_message + "\nQuality of support on a scale from 1 to 3:"
        quality_message = standard_gpt2.complete(next_prompt)[0]
        print('========================')
        print("quality_message:", quality_message)
        try:
            quality_int = int(quality_message.split(':')[-1])
            if quality_int > 3 or quality_int < 1:
                continue
        except Exception as e:
            print(e)
            continue

        if quality_int == 1:
            train_text = quality_message.replace("Support's high quality reply:",  "Support's low quality reply:")
        elif quality_int == 2:
            train_text = quality_message.replace("Support's high quality reply:",  "Support's reply:")
        else:
            train_text = quality_message

        train_gpt2.train(train_text, iteration=i)

        # Log the quality_int value in Visdom and also set a title for the plot.
        vis.line(X=torch.tensor([i]), Y=torch.tensor([quality_int]), win='quality_int', update='append', opts=dict(title='quality_int'))

        # Smooth the quality_int value for logging.
        smoothed_quality_int = 0.99 * smoothed_quality_int + 0.01 * quality_int
        print(f'Smoothed quality_int: {smoothed_quality_int}')

        # Smooth the quality_int for logging using Visdom.
        vis.line(X=torch.tensor([i]), Y=torch.tensor([smoothed_quality_int]), win='smoothed quality_int', update='append', opts=dict(title='smoothed quality_int'))

        i += 1






if __name__ == '__main__':                                                                              
    main()                                                                                              


