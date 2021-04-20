import torch
import torch.nn as nn
import os
import argparse

from tqdm import trange

from dataset import DiscreteLogDataset
from logger import Logger
from utils import accuracy, save_albert, get_lr
from models import AlbertForDiffieHellman

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AlbertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau


import warnings
warnings.filterwarnings("ignore")


def eval(args):
	device = torch.device("cuda")

	validation_iterations = args.val_iterations
	batch_size = args.batch_size
	bits = args.bits

	print("Building Model...")

	# build albert model
	model_config = AlbertConfig(
		vocab_size=2,
		hidden_size=512 * args.albert_scale,
		embedding_size=256,
		intermediate_size=2048 * args.albert_scale,
		num_attention_heads=8 * args.albert_scale,
		max_position_embeddings=2 * bits,
		type_vocab_size=1
	)

	model = AlbertForDiffieHellman(model_config)

	# load pretrained weights
	model.load_state_dict(torch.load(args.eval_path))

	print("Creating Dataloaders...")

	# create dataloaders
	dataset = DiscreteLogDataset(bits=bits)
	dataloader = iter(DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1))

	# set up model as configured
	if args.parallel:
		model = nn.DataParallel(model)
	model.to(device)

	print("Training beginning...")

	torch.cuda.empty_cache()
	# validation step:
	eval_loss = 0
	eval_accuracy = 0
	model.eval()
	with torch.no_grad():
		validation_iterator = trange(int(validation_iterations), desc="Validation")
		for _ in enumerate(validation_iterator):
			inputs, labels = next(dataloader)
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs, labels=labels)
			eval_loss += outputs[0].mean().item()
			eval_accuracy += accuracy(outputs[1].detach().cpu().numpy().round(), labels.detach().cpu().numpy())

	eval_loss /= validation_iterations
	eval_accuracy /= validation_iterations
	message = f"Eval Loss: {eval_loss}\tEval Accuracy: {eval_accuracy}"
	print(message)


def train(args):
	# set up output directory
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	logger = Logger(output_dir + "/out.log")

	device = torch.device("cuda")

	iterations = args.iterations
	validation_iterations = args.val_iterations
	batch_size = args.batch_size
	num_log_save_steps = args.log_save_steps
	learning_rate = args.init_lr
	weight_decay = args.weight_decay
	bits = args.bits

	print("Building Model...")

	# build albert model
	model_config = AlbertConfig(
		vocab_size=2,
		hidden_size=512 * args.albert_scale,
		embedding_size=256,
		intermediate_size=2048 * args.albert_scale,
		num_attention_heads=8 * args.albert_scale,
		max_position_embeddings=2 * bits,
		type_vocab_size=1
	)

	model = AlbertForDiffieHellman(model_config)

	# load pretrained weights
	if args.transfer:
		pretrained_dict = torch.load(args.transfer_path)
		for name, param in pretrained_dict.items():
			if name not in model.state_dict() or model.state_dict()[name].data.shape != param.shape:
				continue
			model.state_dict()[name].data.copy_(param)

	print("Creating Dataloaders...")

	# create dataloaders
	dataset = DiscreteLogDataset(bits=bits)
	dataloader = iter(DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1))

	no_decay = ['bias', 'LayerNorm.weight']

	# create optimizer based on warmup
	if args.warmup:
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.albert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
			{'params': [p for n, p in model.albert.embeddings.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
			{'params': [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
			{'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	else:
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_factor, patience=args.patience)

	# set up model as configured
	model.train()
	if args.parallel:
		model = nn.DataParallel(model)
	model.to(device)

	print("Training beginning...")

	iterator = trange(int(iterations), desc="Iteration")

	for count in iterator:
		# unfreeze transformer if done with warmup
		if args.warmup and count == args.warmup_iterations:
			no_decay = ['bias', 'LayerNorm.weight']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
				{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
			]

			optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
			scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_factor, patience=args.patience)

		# get batch
		inputs, labels = next(dataloader)
		torch.cuda.empty_cache()
		inputs = inputs.to(device)
		labels = labels.to(device)

		# forward pass
		outputs = model(inputs, labels=labels)

		# get loss
		loss = outputs[0].mean()

		# backward pass
		loss.backward()
		optimizer.step()
		model.zero_grad()

		# log and save
		if count % num_log_save_steps == 0:
			torch.cuda.empty_cache()
			# validation step:
			average_loss = 0
			average_accuracy = 0
			model.eval()
			with torch.no_grad():
				validation_iterator = trange(int(validation_iterations), desc="Validation")
				for _ in enumerate(validation_iterator):
					inputs, labels = next(dataloader)
					inputs = inputs.to(device)
					labels = labels.to(device)

					outputs = model(inputs, labels=labels)
					average_loss += outputs[0].mean().item()
					average_accuracy += accuracy(outputs[1].detach().cpu().numpy().round(), labels.detach().cpu().numpy())

			scheduler.step(average_loss)
			model.train()

			average_loss /= validation_iterations
			average_accuracy /= validation_iterations
			# log
			message = f"Loss: {average_loss}\tAccuracy: {average_accuracy}\tIteration: {count}\tLR: {get_lr(optimizer)}"
			logger.log(message)
			# save
			save_dir = f"{output_dir}/checkpoint-{count}"
			save_albert(save_dir, model.module)

	save_albert(f"{output_dir}/final", model.module)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--iterations", type=int, default=1e5)
	parser.add_argument("--val_iterations", type=int, default=1e2)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--log_save_steps", type=int, default=1000)
	parser.add_argument("--init_lr", type=float, default=1e-5)
	parser.add_argument("--weight_decay", type=float, default=.01)
	parser.add_argument("--bits", type=int, default=64)
	parser.add_argument("--transfer", default=False, action="store_true")
	parser.add_argument("--transfer_path", type=str, default="outputs/1/best/model.pt")
	parser.add_argument("--warmup", default=False, action="store_true")
	parser.add_argument("--warmup_iterations", type=int, default=10000)
	parser.add_argument("--decay_factor", type=float, default=.5)
	parser.add_argument("--patience", type=int, default=3)
	parser.add_argument("--albert_scale", type=int, default=4)
	parser.add_argument("--output_dir", type=str, default="outputs/train_64bit")
	parser.add_argument("--parallel", default=False, action="store_true")
	parser.add_argument("--eval", default=False, action="store_true")
	parser.add_argument("--eval_path", type=str, default="")

	args = parser.parse_args()
	eval(args) if args.eval else train(args)
