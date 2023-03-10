#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import copy
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from UNet_RM_Prune_v2 import compute_prune_rate, prune_model

#------------------------------------------------------------------------------
#  Poly learning-rate Scheduler
#------------------------------------------------------------------------------
def poly_lr_scheduler(optimizer, init_lr, curr_iter, max_iter, power=0.9):
	for g in optimizer.param_groups:
		g['lr'] = init_lr * (1 - curr_iter/max_iter)**power


def sparse_mean(sparse_modules):
	sparse_layer_weight_concat = torch.cat(list(map(lambda m: m._conv.weight.view(-1), sparse_modules)))
	assert len(sparse_layer_weight_concat.shape) == 1, "sparse_weight_concat is expected as a vector"
	sparse_weights_mean = sparse_layer_weight_concat.mean()
	return sparse_weights_mean


def _compute_polarization_sparsity(sparse_modules: list, lbd, t, alpha, bn_weights_mean):
	sparsity_loss = 0
	for m in sparse_modules:
		sparsity_term = t * torch.sum(torch.abs(m._conv.weight)) - torch.sum(torch.abs(m._conv.weight - alpha * bn_weights_mean))
		sparsity_loss += lbd * sparsity_term
	return sparsity_loss


def bn_sparsity(model, sparsity=2.5e-5, t=1.0, alpha=0.6):
	sparse_modules = model.get_sparse_layer()
	sparse_weights_mean = sparse_mean(sparse_modules)
	sparse_loss = _compute_polarization_sparsity(sparse_modules, lbd=sparsity, t=t, alpha=alpha, bn_weights_mean=sparse_weights_mean)
	return sparse_loss


def clamp_bn(model):
	sparse_modules = model.get_sparse_layer()
	for m in sparse_modules:
		m._conv.weight.data.clamp_(0, 1)


def limit_conv_weight(model):
	conv_list = model.get_conv_layer()
	for m in conv_list:
		m.weight.data.clamp_(-1, 1)

#------------------------------------------------------------------------------
#   Class of Trainer
#------------------------------------------------------------------------------
class Trainer(BaseTrainer):
	"""
	Trainer class

	Note:
		Inherited from BaseTrainer.
	"""
	def __init__(self, model, loss, metrics, optimizer, resume, config,
				 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, sparsity_factor=0.0):
		super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)	# baseTrainer
		self.config = config
		self.data_loader = data_loader
		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.max_iter = len(self.data_loader) * self.epochs
		self.init_lr = optimizer.param_groups[0]['lr']
		self.sparsity_factor = sparsity_factor
		self.start_prune = 20


	def _eval_metrics(self, output, target):
		acc_metrics = np.zeros(len(self.metrics))
		for i, metric in enumerate(self.metrics):
			acc_metrics[i] += metric(output, target)
		return acc_metrics


	def _train_epoch(self, epoch):
		"""
		Training logic for an epoch

		:param epoch: Current training epoch.
		:return: A log that contains all information you want to save.

		Note:
			If you have additional information to record, for example:
				> additional_log = {"x": x, "y": y}
			merge it with log before return. i.e.
				> log = {**log, **additional_log}
				> return log

			The metrics in log must have the key 'metrics'.
		"""
		print("Train on epoch...")
		self.model.train()
		self.writer_train.set_step(epoch)
		if epoch % self.start_prune == 0:
			reset_metrics = True
		else:
			reset_metrics = False
		# Perform training
		total_loss = 0
		total_metrics = np.zeros(len(self.metrics))
		n_iter = len(self.data_loader)
		for batch_idx, (data, target) in tqdm(enumerate(self.data_loader), total=n_iter):
			curr_iter = batch_idx + (epoch-1)*n_iter
			data, target = data.to(self.device), target.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.loss(output, target)
			sparse_loss = bn_sparsity(self.model)
			# print('sparse_loss: ', sparse_loss)
			loss += sparse_loss
			loss.backward()
			# use MCnet_resnet18
			if self.sparsity_factor > 0:
				self.model.sparsity_BN(self.sparsity_factor)	 # 稀疏化训练

			# lr = self.optimizer.param_groups[0]['lr']
			# print('lr......', lr)
			self.optimizer.step()
			clamp_bn(self.model)
			limit_conv_weight(self.model)

			total_loss += loss.item()
			total_metrics += self._eval_metrics(output, target)

			if (batch_idx==n_iter-2) and (self.verbosity>=2):
				self.writer_train.add_image('train/input', make_grid(data[:,:3,:,:].cpu(), nrow=4, normalize=True))
				self.writer_train.add_image('train/label', make_grid(target.unsqueeze(1).cpu(), nrow=4, normalize=True))
				if type(output)==tuple or type(output)==list:
					self.writer_train.add_image('train/output', make_grid(F.softmax(output[0], dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))
				else:
					# self.writer_train.add_image('train/output', make_grid(output.cpu(), nrow=4, normalize=True))
					self.writer_train.add_image('train/output', make_grid(F.softmax(output, dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))

			# poly_lr_scheduler(self.optimizer, self.init_lr, curr_iter, self.max_iter, power=0.9)

		# Record log
		total_loss /= len(self.data_loader)
		total_metrics /= len(self.data_loader)
		log = {
			'train_loss': total_loss,
			'train_metrics': total_metrics.tolist(),
		}

		# Write training result to TensorboardX
		self.writer_train.add_scalar('loss', total_loss)
		for i, metric in enumerate(self.metrics):
			self.writer_train.add_scalar('metrics/%s'%(metric.__name__), total_metrics[i])

		if self.verbosity>=2:
			for i in range(len(self.optimizer.param_groups)):
				self.writer_train.add_scalar('lr/group%d'%(i), self.optimizer.param_groups[i]['lr'])

		# Perform validating
		if self.do_validation:
			print("Validate on epoch...")
			val_log = self._valid_epoch(epoch)
			log = {**log, **val_log}

		# Learning rate scheduler
		if self.lr_scheduler is not None:
			self.lr_scheduler.step()

		return log, reset_metrics


	def _valid_epoch(self, epoch):
		"""
		Validate after training an epoch

		:return: A log that contains information about validation

		Note:
			The validation metrics in log must have the key 'valid_metrics'.
		"""
		self.model.eval()
		total_val_loss = 0
		total_val_metrics = np.zeros(len(self.metrics))
		n_iter = len(self.valid_data_loader)
		self.writer_valid.set_step(epoch)

		with torch.no_grad():
			# Validate
			for batch_idx, (data, target) in tqdm(enumerate(self.valid_data_loader), total=n_iter):
				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				loss = self.loss(output, target)

				total_val_loss += loss.item()
				total_val_metrics += self._eval_metrics(output, target)

				if (batch_idx==n_iter-2) and(self.verbosity>=2):
					self.writer_valid.add_image('valid/input', make_grid(data[:,:3,:,:].cpu(), nrow=4, normalize=True))
					self.writer_valid.add_image('valid/label', make_grid(target.unsqueeze(1).cpu(), nrow=4, normalize=True))
					if type(output)==tuple or type(output)==list:
						self.writer_valid.add_image('valid/output', make_grid(F.softmax(output[0], dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))
					else:
						# self.writer_valid.add_image('valid/output', make_grid(output.cpu(), nrow=4, normalize=True))
						self.writer_valid.add_image('valid/output', make_grid(F.softmax(output, dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))

			# Record log
			total_val_loss /= len(self.valid_data_loader)
			total_val_metrics /= len(self.valid_data_loader)
			model_cpu = copy.deepcopy(self.model)
			model_cpu = model_cpu.to('cpu')
			_, model_prune = prune_model(model_cpu)
			prune_rate = compute_prune_rate(model_cpu, model_prune)
			val_log = {
				'valid_loss': total_val_loss,
				'valid_metrics': total_val_metrics.tolist(),
				'prune_rate': prune_rate
			}

			# Write validating result to TensorboardX
			self.writer_valid.add_scalar('loss', total_val_loss)
			for i, metric in enumerate(self.metrics):
				self.writer_valid.add_scalar('metrics/%s'%(metric.__name__), total_val_metrics[i])

		return val_log
