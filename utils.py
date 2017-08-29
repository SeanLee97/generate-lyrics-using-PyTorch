# !/usr/bin/env python
# -*- coding: utf-8 -*-

import time, math
import os, glob
import torch
import numpy as np

def time_since(since):
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m*60
	return '%dm %ds' % (m, s)

def save_model(model, epoch, max_keep=5):
	if not os.path.exists('./runtime'):
		os.makedirs('runtime')
	f_list = glob.glob(os.path.join('./runtime', 'model') + '-*.ckpt')
	if len(f_list) >= max_keep + 2:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		to_delete = [f_list[i] for i in np.argsort(epoch_list)[-max_keep:]]
		for f in to_delete:
			os.remove(f)
	name = 'model-{}.ckpt'.format(epoch)
	file_path = os.path.join('./runtime', name)
	torch.save(model.state_dict(), file_path)

def load_previous_model(model):
	f_list = glob.glob(os.path.join('./runtime', 'model') + '-*.ckpt')
	start_epoch = 1
	if len(f_list) >= 1:
		epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
		last_checkpoint = f_list[np.argmax(epoch_list)]
		if os.path.exists(last_checkpoint):
			print('load from {}'.format(last_checkpoint))
			model.load_state_dict(torch.load(last_checkpoint))
			start_epoch = np.max(epoch_list)
	return model, start_epoch
