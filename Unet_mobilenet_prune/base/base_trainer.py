#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
from time import time
import os, math, json, logging, datetime, torch
from utils.visualization import WriterTensorboardX
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from models.backbonds import MobileNetV2
#------------------------------------------------------------------------------
#   Class of BaseTrainer
#------------------------------------------------------------------------------
class BaseTrainer:
	"""
	Base class for all trainers
	"""
	def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
		self.config = config
		# Setup directory for checkpoint saving
		start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
		self.img_save_dir = config['trainer']['save_dir']
		self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
		os.makedirs(self.checkpoint_dir, exist_ok=True)

		# Setup logger
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s %(message)s",
			handlers=[
				logging.FileHandler(os.path.join(self.checkpoint_dir, "train.log")),
				logging.StreamHandler(),
				])
		self.logger = logging.getLogger(self.__class__.__name__)

		# Setup GPU device if available, move model into configured device
		self.device, device_ids = self._prepare_device(config['n_gpu'])
		self.model = model.to(self.device)
		if len(device_ids) > 1:
			self.model = torch.nn.DataParallel(model, device_ids=device_ids)

		self.loss = loss
		self.metrics = metrics
		self.optimizer = optimizer

		self.epochs = config['trainer']['epochs']
		self.save_freq = config['trainer']['save_freq']
		self.verbosity = config['trainer']['verbosity']

		self.train_logger = train_logger

		# configuration to monitor model performance and save best
		self.monitor = config['trainer']['monitor']
		self.monitor_mode = config['trainer']['monitor_mode']
		assert self.monitor_mode in ['min', 'max', 'off']
		self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
		self.start_epoch = 1

		# setup visualization writer instance
		writer_train_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time, "train")
		writer_valid_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time, "valid")
		self.writer_train = WriterTensorboardX(writer_train_dir, self.logger, config['visualization']['tensorboardX'])
		self.writer_valid = WriterTensorboardX(writer_valid_dir, self.logger, config['visualization']['tensorboardX'])

		# Save configuration file into checkpoint directory
		config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
		with open(config_save_path, 'w') as handle:
			json.dump(config, handle, indent=4, sort_keys=False)

		# Resume
		if resume:
			self._resume_checkpoint(resume)
	

	def _prepare_device(self, n_gpu_use):
		""" 
		setup GPU device if available, move model into configured device
		""" 
		n_gpu = torch.cuda.device_count()
		if n_gpu_use > 0 and n_gpu == 0:
			self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
			n_gpu_use = 0
		if n_gpu_use > n_gpu:
			msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
			self.logger.warning(msg)
			n_gpu_use = n_gpu
		device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
		list_ids = list(range(n_gpu_use))
		return device, list_ids


	def train(self):
		train_iou = [0 for _ in range(self.epochs + 1)]
		train_B_iou = [0 for _ in range(self.epochs + 1)]
		train_F_iou = [0 for _ in range(self.epochs + 1)]
		val_iou = [0 for _ in range(self.epochs + 1)]
		val_B_iou = [0 for _ in range(self.epochs + 1)]
		val_F_iou = [0 for _ in range(self.epochs + 1)]
		train_loss = [0 for _ in range(self.epochs + 1)]
		train_B_loss = [0 for _ in range(self.epochs + 1)]
		train_F_loss = [0 for _ in range(self.epochs + 1)]
		val_loss = [0 for _ in range(self.epochs + 1)]
		val_B_loss = [0 for _ in range(self.epochs + 1)]
		val_F_loss = [0 for _ in range(self.epochs + 1)]
		epoch_num = [i for i in range(self.epochs + 1)]
		writer = SummaryWriter(
			logdir='/media/byd/A264AC9264AC6AAD/DataSet/8_25_data/unet_mobilenet_prune/tensorboard')
		for epoch in range(self.start_epoch, self.epochs + 1):
			for name, sub_module in self.model.named_modules():
				if isinstance(sub_module, MobileNetV2.SparseGate):
					writer.add_histogram("sparse_gate/" + name, sub_module._conv.weight,epoch)
			self.logger.info("\n----------------------------------------------------------------")
			self.logger.info("[EPOCH %d]" % (epoch))
			start_time = time()
			result, reset_metrics = self._train_epoch(epoch)
			a = self.optimizer
			print('LR: ', a.param_groups[0]['lr'])

			with open('lr.txt', 'a') as ff:
				ff.write(str(epoch) + ' ' + str(a.param_groups[0]['lr']) + '\n')
			finish_time = time()
			self.logger.info("Finish at {}, Runtime: {:.3f} [s]".format(datetime.datetime.now(), finish_time-start_time))
			
			# save logged informations into log dict
			log = {}
			log_dir = os.path.join(self.img_save_dir, 'log')
			os.makedirs(log_dir, exist_ok=True)
			for key, value in result.items():
				if key == 'train_metrics':
					log.update({'train_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
				elif key == 'valid_metrics':
					log.update({'valid_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
				else:
					log[key] = value

			for keyy, valuee in log.items():
				if keyy == 'train_total_loss':
					train_loss[epoch-1] = valuee
				elif keyy == 'train_background_loss':
					train_B_loss[epoch - 1] = valuee
				elif keyy == 'train_foreground_loss':
					train_F_loss[epoch - 1] = valuee
				elif keyy == 'train_miou':
					train_iou[epoch - 1] = valuee
				elif keyy == 'train_miou_background':
					train_B_iou[epoch - 1] = valuee[0]
				elif keyy == 'train_miou_foreground':
					train_F_iou[epoch - 1] = valuee[0]
				elif keyy == 'valid_loss':
					val_loss[epoch - 1] = valuee
				elif keyy == 'val_background_loss':
					val_B_loss[epoch - 1] = valuee
				elif keyy == 'val_foreground_loss':
					val_F_loss[epoch - 1] = valuee
				elif keyy == 'valid_miou':
					val_iou[epoch - 1] = valuee
				elif keyy == 'val_miou_background':
					val_B_iou[epoch - 1] = valuee[0]
				elif keyy == 'val_miou_foreground':
					val_F_iou[epoch - 1] = valuee[0]

				log_txt = open(log_dir + 'log.txt', 'a')
				log_txt.write(str(epoch) + '  ' + str(keyy) + '  ' + str(valuee) + '\n')
				log_txt.close()

			# 画图
			styles = plt.style.available
			linear_Clip = [13, 18, 22, 30, 36]
			input_values = [1, 2, 4, 8, 16]
			fig, ax = plt.subplots()
			ax.plot(epoch_num, train_loss, linewidth=1, label='train_loss', color='#9B59B6')
			ax.plot(epoch_num, train_B_loss, linewidth=1, label='train_B_loss', color='#000000')
			ax.plot(epoch_num, train_F_loss, linewidth=1, label='train_F_loss', color='#0000FF')
			ax.plot(epoch_num, val_loss, linewidth=1, label='val_loss', color='#E67C1F')
			ax.plot(epoch_num, val_B_loss, linewidth=1, label='val_B_loss', color='#808080')
			ax.plot(epoch_num, val_F_loss, linewidth=1, label='val_F_loss', color='#FFB6C1')
			ax.set_ylim(0, 1.6)
			plt.subplots_adjust(bottom=0.10)
			# 设置背景颜色
			plt.rcParams['axes.facecolor'] = 'white'
			# 设置背景线条样式
			plt.grid(linestyle='--')
			plt.xlabel('epoch')  # X轴标签
			plt.ylabel("loss")  # Y轴标签
			plt.margins(0.15)
			plt.legend()
			plt.savefig(log_dir + "loss.png")

			fig1, ax1 = plt.subplots()
			ax1.plot(epoch_num, train_iou, linewidth=1, label='train_miou', color='#9B59B6')
			ax1.plot(epoch_num, train_B_iou, linewidth=1, label='train_B_iou', color='#000000')
			ax1.plot(epoch_num, train_F_iou, linewidth=1, label='train_F_iou', color='#0000FF')
			ax1.plot(epoch_num, val_iou, linewidth=1, label='val_miou', color='#E67C1F')
			ax1.plot(epoch_num, val_B_iou, linewidth=1, label='val_B_iou', color='#808080')
			ax1.plot(epoch_num, val_F_iou, linewidth=1, label='val_F_iou', color='#FFB6C1')
			ax1.set_ylim(0, 1)
			plt.subplots_adjust(bottom=0.10)
			# 设置背景颜色
			plt.rcParams['axes.facecolor'] = 'white'
			# 设置背景线条样式
			plt.grid(linestyle='--')
			plt.xlabel('epoch')  # X轴标签
			plt.ylabel("miou")  # Y轴标签
			plt.margins(0.15)
			plt.legend()
			plt.savefig(log_dir + "miou.png")

			# print logged informations to the screen
			if self.train_logger is not None:
				self.train_logger.add_entry(log)
				if self.verbosity >= 1:
					for key, value in sorted(list(log.items())):
						self.logger.info('{:25s}: {}'.format(str(key), value))

			# evaluate model performance according to configured metric, save best checkpoint as model_best

			best = False
			if self.monitor_mode != 'off':
				try:
					if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
						(self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
						self.logger.info("Monitor improved from %f to %f" % (self.monitor_best, log[self.monitor]))
						self.monitor_best = log[self.monitor]
						best = True
				except KeyError:
					if epoch == 1:
						msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
							+ "for performance monitoring. model_best checkpoint won\'t be updated."
						self.logger.warning(msg)
			if reset_metrics:
				self.monitor_best = log[self.monitor]
				best = True
			# Save checkpoint
			self._save_checkpoint(epoch, save_best=best)


	def _train_epoch(self, epoch):
		"""
		Training logic for an epoch

		:param epoch: Current epoch number
		"""
		raise NotImplementedError


	def _save_checkpoint(self, epoch, save_best=False):
		"""
		Saving checkpoints

		:param epoch: current epoch number
		:param log: logging information of the epoch
		:param save_best: if True, rename the saved checkpoint to 'model_best.pth'
		"""
		# Construct savedict
		arch = type(self.model).__name__
		state = {
			'arch': arch,
			'epoch': epoch,
			'logger': self.train_logger,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'monitor_best': self.monitor_best,
			'config': self.config
		}

		# Save checkpoint for each epoch
		if self.save_freq is not None:	# Use None mode to avoid over disk space with large models
			if epoch % self.save_freq == 0:
				filename = os.path.join(self.checkpoint_dir, 'epoch{}.pth'.format(epoch))
				torch.save(state, filename)
				self.logger.info("Saving checkpoint at {}".format(filename))

		# Save the best checkpoint
		if save_best:
			best_path = os.path.join(self.checkpoint_dir, '{}.pth'.format(epoch))
			torch.save(state, best_path)
			self.logger.info("Saving current best at {}".format(best_path))
		else:
			self.logger.info("Monitor is not improved from %f" % (self.monitor_best))


	def _resume_checkpoint(self, resume_path):
		"""
		Resume from saved checkpoints

		:param resume_path: Checkpoint path to be resumed
		"""
		self.logger.info("Loading checkpoint: {}".format(resume_path))
		checkpoint = torch.load(resume_path)
		self.start_epoch = checkpoint['epoch'] + 1
		self.monitor_best = checkpoint['monitor_best']

		# load architecture params from checkpoint.
		if checkpoint['config']['arch'] != self.config['arch']:
			self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
								'This may yield an exception while state_dict is being loaded.')
		self.model.load_state_dict(checkpoint['state_dict'], strict=True)

		# # load optimizer state from checkpoint only when optimizer type is not changed. 
		# if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
		# 	self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
		# 						'Optimizer parameters not being resumed.')
		# else:
		# 	self.optimizer.load_state_dict(checkpoint['optimizer'])
	
		self.train_logger = checkpoint['logger']
		self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch-1))