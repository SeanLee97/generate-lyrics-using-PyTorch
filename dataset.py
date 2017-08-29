# !/usr/bin/env python
# -*- coding: utf-8 -*-
import string
import random
import torch
from torch.autograd import Variable
from config import *

class Lang(object):
	def __init__(self, filename):
		self.char2idx = {}
		self.idx2char = {}
		self.n_words = 0
		self.process(filename)

	def process(self, filename):
		with open(filename, 'r') as f:
			for line in f.readlines():
				words = set(line)
				comm = words & set(self.char2idx)
				for word in words:
					if word not in comm:
						self.char2idx[word] = self.n_words
						self.idx2char[self.n_words] = word
						self.n_words += 1

class Dataset(object):
	def __init__(self, filename):
		self.lang = Lang(filename)
		#print(self.lang.idx2char)
		self.data = self.load_file(filename)

	def load_file(self, filename):
		data = []
		with open(filename, 'r') as f:
			data = f.read()
		return data
	
	def random_chunk(self, chunk_len = CHUNK_LEN):
		start_idx = random.randint(0, len(self.data) - chunk_len)
		end_idx = start_idx + chunk_len + 1
		return self.data[start_idx:end_idx]
	
	def get_variable(self, string):
		tensor = torch.zeros(len(string)).long()  # FloatTensor->LongTensor
		for c in range(len(string)):
			tensor[c] = self.lang.char2idx[string[c]]
		return Variable(tensor)

	def random_training_set(self):
		chunk = self.random_chunk()
		
		'''
		print("chunk> ", chunk)
		print("--------------")
		print("chunk[:-1]> ", chunk[:-1])
		print("--------------")
		print("chunk[1:]> ", chunk[1:])	
		exit()
		'''
		input = self.get_variable(chunk[:-1])
		target = self.get_variable(chunk[1:])
		return input, target

