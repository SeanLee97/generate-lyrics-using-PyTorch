# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers = 1, dropout_p=0.1):
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
			
		self.encoder = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
		self.dropout = nn.Dropout(dropout_p)
		self.decoder = nn.Linear(hidden_size, output_size)

	def forward(self, input, hidden):
		# input.view(1, -1) size 1 x input_size
		# input size 1 x 1 x hidden_size
		input = self.encoder(input.view(1, -1))
		input = self.dropout(input)
		# input.view(1 ,1 ,-1) size seq_len x batch x input_size => 1 x 1 x input_size
		# hidden size n_layers x batch x hidden_size => n_layers x 1 x input_size
		# output size seq_len x batch x hidden_size * num_directions => 1 x 1 x hidden_size
		output, hidden = self.gru(input.view(1, 1, -1), hidden)
		# output.view(1, -1) size 1 x (1 x hidden_size)
		# output size 1 x output_size
		output = self.decoder(output.view(1, -1))
		return output, hidden
	def init_hidden(self):
		# size self.n_layers x 1 x self.hidden_size
		return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
