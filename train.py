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
		# 多项分布随机采样
		output_dist = output.data.view(-1).div(temperature).exp() # exp()保证各项均为正数
		top_i = torch.multinomial(output_dist, 1)[0] # int

		predicted_char = dataset.lang.idx2char[top_i] 
		predicted += predicted_char
		input = dataset.get_variable(predicted_char)
	return predicted	

def train(model, optimizer, loss_fn, dataset, start_epoch=1):
	start = time.time()
	loss_avg = 0
	for epoch in range(start_epoch, N_EPOCHS+1):
		input, target = dataset.random_training_set()
		hidden = model.init_hidden()
		optimizer.zero_grad()
		loss = 0
		for c in range(CHUNK_LEN):
			output, hidden = model(input[c], hidden)
			loss += loss_fn(output, target[c])
		loss.backward()
		optimizer.step()
		each_loss_avg = loss.data[0] / CHUNK_LEN
		loss_avg += each_loss_avg
		if epoch % PRINT_EVERY == 0:
			print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch/N_EPOCHS*100, each_loss_avg))
			print(evaluate(model, dataset, '你要', 100),'\n')
			save_model(model, epoch)

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
	train(model, optimizer, loss_fn, dataset, start_epoch=start_epoch)
if __name__ == '__main__':
	main()
