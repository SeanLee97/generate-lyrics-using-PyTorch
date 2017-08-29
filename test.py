# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import string
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from config import *
from model import RNN
from dataset import Dataset
from utils import *

def evaluate(model, dataset, prime_str='我只', predict_len = 100, temperature = 0.8):
	hidden = model.init_hidden()
	prime_input = dataset.get_variable(prime_str)
	predicted = prime_str
	for p in range(len(prime_str)-1):
		_, hidden = model(prime_input[p], hidden)
	input = prime_input[-1]
	
	for p in range(predict_len):
		output, hidden = model(input, hidden)
		output_dist = output.data.view(-1).div(temperature).exp()
		top_i = torch.multinomial(output_dist, 1)[0]	
		predicted_char = dataset.lang.idx2char[top_i] 
		predicted += predicted_char
		input = dataset.get_variable(predicted_char)
	return predicted	

def generate(model, dataset, word, gen_len):
	print("gen> ", evaluate(model, dataset, word, gen_len))

def main():
	path = './runtime/data.pkl'
	if not os.path.exists(path):
		with open(path, 'wb') as f:
			dataset = Dataset('./data/lyrics.txt')
			pickle.dump(dataset, f)
	else:
		with open(path, 'rb') as f:
			dataset = pickle.load(f)
	model = RNN(dataset.lang.n_words, HIDDEN_SIZE, dataset.lang.n_words, N_LAYERS)	
	model, start_epoch = load_previous_model(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.CrossEntropyLoss()
	while True:
		word = input("input> ")
		if len(word) < 2:
			print("输入两个字以上哦！")
			continue
		gen_len = input("length> ")
		generate(model, dataset, word, int(gen_len))
	
if __name__ == '__main__':
	main()
