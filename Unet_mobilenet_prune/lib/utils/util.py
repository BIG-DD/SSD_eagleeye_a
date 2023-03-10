import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from contextlib import contextmanager
import re
import onnx
import onnxruntime as ort
import onnxsim
import copy


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def create_logger(cfg, cfg_path, phase='train', rank=-1):
    # set up logger dir
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model_name = cfg.MODEL.NAME
    cfg_path = os.path.basename(cfg_path).split('.')[0]

    if rank in [-1, 0]:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}_{}.log'.format(cfg_path, time_str, phase)
        # set up tensorboard_log_dir
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model_name / \
                                  (cfg_path + '_' + time_str + '_' + phase)
        final_output_dir = tensorboard_log_dir
        if not tensorboard_log_dir.exists():
            print('=> creating {}'.format(tensorboard_log_dir))
            tensorboard_log_dir.mkdir(parents=True)

        final_log_file = tensorboard_log_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        return None, None, None


def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            #model.parameters(),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999)
        )

    return optimizer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(epoch, name, model, optimizer, output_dir, filename, is_best=False):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def load_model_parameter(rank, cfg, logger, model, optimizer, val_loader, begin_epoch, Encoder_para_idx, Det_Head_para_idx, Ll_Seg_Head_para_idx=None):
    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            # cfg.NEED_AUTOANCHOR = False     #disable autoanchor

        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0, 25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))

        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            # cfg.NEED_AUTOANCHOR = False     #disable autoanchor

        if cfg.TRAIN.SEG_ONLY:  # Only train two segmentation branchs
            logger.info('freeze encoder and Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.DET_ONLY:  # Only train detection branch
            logger.info('freeze encoder and two Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_SEG_ONLY:  # Only train encoder and two segmentation branchs
            logger.info('freeze Det head...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:  # Only train encoder and detection branchs
            logger.info('freeze two Seg heads...')
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Ll_Seg_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

        if cfg.TRAIN.LANE_ONLY:
            logger.info('freeze encoder and Det head and Da_Seg heads...')
            # print(model.named_parameters)
            for k, v in model.named_parameters():
                v.requires_grad = True  # train all layers
                if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
                    print('freezing %s' % k)
                    v.requires_grad = False

    return model, optimizer, begin_epoch


def write_onnx_model(cfg, model, opset_version=11, filename='checkpoint.onnx', output_names='det_out'):
    filename = os.path.splitext(filename)[0] + '.onnx'
    model.eval()
    is_cuda = next(model.parameters()).is_cuda
    dummy_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]))
    dummy_input = dummy_input.cuda() if is_cuda else dummy_input

    torch.onnx.export(model, dummy_input, filename,
                      verbose=False, opset_version=opset_version, input_names=['images'],
                      output_names=[output_names])

    print('convert', filename, 'to onnx finish!!!')
    # Checks requirement pytorch==1.7.0 torchvision==0.8.0
    model_onnx = onnx.load(filename)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))  # print
    do_simplify = True
    if do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, 'assert check failed'
        onnx.save(model_onnx, filename)
    try:
        sess = ort.InferenceSession(filename)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e


def measures_model(model, cfg, logger, use_cuda=False, n_measures=10):
    print(model)
    model.train()
    from torchsummary import summary
    total_params, trainable_params, non_trainable_params, total_input_size, total_output_size, total_params_size, total_size = summary(model, (3, 384, 384), device='cpu')

    logger.info("================================================================")
    logger.info("Total params: {0:,}".format(total_params))
    logger.info("Trainable params: {0:,}".format(trainable_params))
    logger.info("Non-trainable params: {0:,}".format(non_trainable_params))
    logger.info("----------------------------------------------------------------")
    logger.info("Input size (MB): %0.2f" % total_input_size)
    logger.info("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger.info("Params size (MB): %0.2f" % total_params_size)
    logger.info("Estimated Total Size (MB): %0.2f" % total_size)
    logger.info("----------------------------------------------------------------")

    from time import time
    # ------------------------------------------------------------------------------
    #   Measure time
    # ------------------------------------------------------------------------------
    input = torch.randn([1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]], dtype=torch.float)
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        model.cuda()
        input = input.cuda()

    for _ in range(10):
        model(input)

    start_time = time()
    for _ in range(n_measures):
        model(input)
    finish_time = time()

    if use_cuda:
        # print("Inference time on cuda: %.2f [ms]" % ((finish_time - start_time) * 1000 / n_measures))
        # print("Inference fps on cuda: %.2f [fps]" % (1 / ((finish_time - start_time) / n_measures)))
        logger.info("Inference time on cuda: %.2f [ms]" % ((finish_time - start_time) * 1000 / n_measures))
        logger.info("Inference fps on cuda: %.2f [fps]" % (1 / ((finish_time - start_time) / n_measures)))
    else:
        # print("Inference time on cpu: %.2f [ms]" % ((finish_time - start_time) * 1000 / n_measures))
        # print("Inference fps on cpu: %.2f [fps]" % (1 / ((finish_time - start_time) / n_measures)))
        logger.info("Inference time on cpu: %.2f [ms]" % ((finish_time - start_time) * 1000 / n_measures))
        logger.info("Inference fps on cpu: %.2f [fps]" % (1 / ((finish_time - start_time) / n_measures)))


# => m_block_cfg 'YOLOV5S_2head'
# load model to device
# Inference time on cpu: 60.67 [ms]
# Inference fps on cpu: 16.48 [fps]

# => m_block_cfg 'YOLOV5S_mobilenetV2_quarter_2head'
# load model to device
# Inference time on cpu: 78.51 [ms]
# Inference fps on cpu: 12.74 [fps]

# => m_block_cfg 'YOLOV5S_mobilenetV2_half_2head'
# load model to device
# Inference time on cpu: 84.38 [ms]
# Inference fps on cpu: 11.85 [fps]

# => m_block_cfg 'YOLOV5S_mobilenetV2_1_2head'
# load model to device
# Inference time on cpu: 112.62 [ms]
# Inference fps on cpu: 8.88 [fps]