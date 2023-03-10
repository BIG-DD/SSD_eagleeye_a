import onnx
import onnxruntime as ort
import onnxsim
import torch
from torch.nn import init
import os
import torch.nn as nn



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def save_checkpoint(epoch, name, model, output_dir, filename, is_best=False, optimizer=None):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict() if optimizer else None,
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def pth2onnx(model, onnx_path, height, width):
    do_simplify = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to('cpu')
    model.eval()

    # Input
    inputs = torch.randn(1, 3, height, width)
    torch.onnx.export(model,
                      inputs,
                      onnx_path,
                      verbose=False,
                      opset_version=11,
                      input_names=['data'],
                      output_names=['seg_out'])
    print('convert', onnx_path, 'to onnx finish!!!')

    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)

    try:
        sess = ort.InferenceSession(onnx_path)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e
