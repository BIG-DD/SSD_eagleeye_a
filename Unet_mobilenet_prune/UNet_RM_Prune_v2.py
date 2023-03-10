# ------------------------------------------------------------------------------
#	Libraries
# ------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d
from models.UNet import MCnet_resnet18, MCnet_resnet18_RM_Prune, MCnet_resnet18_RM
from models.RM_Prune import ResBlock, Bottleneck_mask_RM, get_prune_threshold, Bottleneck2Bottleneck_RM_prune, Conv2d, \
    Conv2Conv_prune, ConvBNReLU, ConvBN
from models.RM_Prune import ConvBN_mask, ConvBNReLU_mask, block2block_RM_prune, Concat, ConvBNReLU2ConvBNReLU_RM_prune, \
    ConvBN2ConvBN_RM_prune, ConvBNReLU_prune, InvertedResidualExp_prune, InvertedResidualNoRes_prune, InvertedResidualRes_prune, \
    ConvTranspose2d_prune, ConvBN_prune, InvertedResidualResConcat_prune
from models.util import save_checkpoint, pth2onnx
import logging
import json
import dataloaders.dataloader as module_data
from torchvision.utils import make_grid
import evaluation.metrics as module_metric
import evaluation.losses as module_loss
from models.backbonds.MobileNetV2 import InvertedResidualExp, InvertedResidualNoRes, InvertedResidualRes, SparseGate, InvertedResidualResConcat
from torchsummary import summary

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def eval_metrics(metrics, output, target):
    acc_metrics = np.zeros(len(metrics))
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics


def valid(model, valid_data_loader, metrics, criterion, device='cuda'):
    """
    Validate after training an epoch

    :return: A log that contains information about validation

    Note:
        The validation metrics in log must have the key 'valid_metrics'.
    """
    model.to(device)
    model.eval()
    total_val_loss = 0
    total_val_metrics = np.zeros(len(metrics))
    n_iter = len(valid_data_loader)
    # writer_valid.set_step(epoch)

    with torch.no_grad():
        # Validate
        for batch_idx, (data, target) in tqdm(enumerate(valid_data_loader), total=n_iter):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_val_loss += loss.item()
            total_val_metrics += eval_metrics(metrics, output, target)

            # if (batch_idx==n_iter-2) and(self.verbosity>=2):
            #     self.writer_valid.add_image('valid/input', make_grid(data[:,:3,:,:].cpu(), nrow=4, normalize=True))
            #     self.writer_valid.add_image('valid/label', make_grid(target.unsqueeze(1).cpu(), nrow=4, normalize=True))
            #     if type(output)==tuple or type(output)==list:
            #         self.writer_valid.add_image('valid/output', make_grid(F.softmax(output[0], dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))
            #     else:
            #         # self.writer_valid.add_image('valid/output', make_grid(output.cpu(), nrow=4, normalize=True))
            #         self.writer_valid.add_image('valid/output', make_grid(F.softmax(output, dim=1)[:,1:2,:,:].cpu(), nrow=4, normalize=True))

        # Record log
        total_val_loss /= len(valid_data_loader)
        total_val_metrics /= len(valid_data_loader)
        val_log = {
            'valid_loss': total_val_loss,
            'valid_metrics': total_val_metrics.tolist(),
        }
        print(val_log)
        # Write validating result to TensorboardX
        # self.writer_valid.add_scalar('loss', total_val_loss)
        # for i, metric in enumerate(metrics):
        #     self.writer_valid.add_scalar('metrics/%s'%(metric.__name__), total_val_metrics[i])

    return val_log

def get_thr(block):
    threshold_list = []
    for m in block.modules():
        if isinstance(m, SparseGate):
            spars_weight = m._conv.weight.view(-1)
            hist_y, hist_x = np.histogram(spars_weight.detach().numpy(), bins=100, range=(0, 1))
            hist_y_diff = np.diff(hist_y)
            for i in range(len(hist_y_diff) - 1):
                if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
                    threshold = hist_x[i + 1]
                    if threshold > 0.2:
                        print(f"WARNING: threshold might be too large: {threshold}")
                    threshold_list.append(threshold)
                    break
    return threshold_list

def prune_model(net):
    net_params = []  # RM+prune后每个模块的网络参数
    # threshold = get_prune_threshold(copy.deepcopy(net), prune_scale_rate=prune_scale_rate)
    threshold = 1e-3
    features = []  # RM+prune re
    block_out_mask = []  # 每个模块输出的网络参数
    concat_num = []  # concat连接的网络层
    in_mask = torch.ones(3) > 0
    for i, block in enumerate(net.model):
        # print("block", block)
        block_prue = None
        if isinstance(block, ConvBNReLU):
            print('ConvBNReLU')
            block_prue, in_mask, block_params = ConvBNReLU_prune(block, in_mask)
            features.append(block_prue)
        elif isinstance(block, InvertedResidualExp):
            print('InvertedResidualExp')
            thr = get_thr(block)
            block_prue, in_mask, block_params = InvertedResidualExp_prune(block, in_mask, thr)
            features.append(block_prue)
        elif isinstance(block, InvertedResidualNoRes):
            print("InvertedResidualNoRes")
            thr = get_thr(block)
            block_prue, in_mask, block_params = InvertedResidualNoRes_prune(block, in_mask, thr, concat_num,
                                                                            block_out_mask)
            features.append(block_prue)
        elif isinstance(block, InvertedResidualRes):
            print("InvertedResidualRes")
            thr = get_thr(block)
            block_prue, in_mask, block_params = InvertedResidualRes_prune(block, in_mask, thr)
            features.append(block_prue)
        elif isinstance(block, InvertedResidualResConcat):
            print('InvertedResidualResConcat')
            thr = get_thr(block)
            block_prue, in_mask, block_params = InvertedResidualResConcat_prune(block, in_mask, thr)
            features.append(block_prue)
        elif isinstance(block, Conv2d):
            print('Conv2d')
            block_prue, in_mask, block_params = Conv2Conv_prune(block, in_mask)
            features.append(block_prue)
        elif isinstance(block, ConvBN):
            block_prue, in_mask, block_params = ConvBN_prune(block, in_mask)
            features.append(block_prue)
        elif isinstance(block, ConvTranspose2d):
            print("ConvTranspose2d")
            block_prue, in_mask, block_params = ConvTranspose2d_prune(block, in_mask)
            features.append(block_prue)
        elif isinstance(block, Concat):
            if i == 13:
                concat_num.append(6)
            elif i == 17:
                concat_num.append(3)
            elif i == 21:
                concat_num.append(1)
            features.append(block)
            block_params = []
        else:
            features.append(block)
            block_params = []

        block_out_mask.append(in_mask)
        net_params.append(block_params)
        if block_prue != None:
            print("block_prue", block_prue)
    model = MCnet_resnet18_RM(net_params)
    return features, model

def compute_prune_rate(model, prune_model):
    base_model = copy.deepcopy(model)
    base_model = base_model.to('cuda')
    summary(base_model, (3, 256, 256))
    base_flops = compute_conv_flops(base_model, cuda=True)
    save_model = copy.deepcopy(prune_model)
    save_model = save_model.to('cuda')
    summary(save_model, (3, 256, 256))
    saved_flops = compute_conv_flops(save_model, cuda=True)
    prune_rate = saved_flops / base_flops
    return prune_rate

# RM
def RM_layer(net, onnx_path):
    features, model = prune_model(net)
    base_model = MCnet_resnet18()
    prune_rate = compute_prune_rate(base_model, model)
    print('prune_rate: ', prune_rate)

    net.model = nn.Sequential(*features)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    save_checkpoint(
        epoch=0,
        name='rm_prune',
        model=net,
        optimizer=optimizer,
        output_dir='./',
        filename='rm_prune.pth'
    )

    checkpoint = torch.load('rm_prune.pth')
    model.load_state_dict(checkpoint['state_dict'])
    #
    config = './config/config_UNet_mobilenet_prune.json'
    config = json.load(open(config))
    valid_loader = get_instance(module_data, 'valid_loader', config).loader
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    criterion = getattr(module_loss, config['loss'])
    valid(model, valid_loader, metrics, criterion, device='cuda')

    pth2onnx(model, onnx_path, height=256, width=256)


def compute_conv_flops(model: torch.nn.Module, cuda=False):
    """
    compute the FLOPs for MobileNet v2 model
    NOTE: ONLY compute the FLOPs for Convolution layers

    if cuda mode is enabled, the model must be transferred to gpu!
    """

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)

        flops = kernel_ops * output_channels * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement()

        flops = weight_ops
        list_linear.append(flops)

    def add_hooks(net, hook_handles: list):
        """
        apply FLOPs handles to conv layers recursively
        """
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            add_hooks(c, hook_handles)

    handles = []
    add_hooks(model, handles)
    demo_input = torch.rand(8, 3, 256, 256)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops


if __name__ == "__main__":
    checkpoint = '/media/byd/A264AC9264AC6AAD/DataSet/12_7_data/eps_1e1_cat/parking_slot/0104_114514/45.pth'
    onnx_path = checkpoint.replace('.pth', '.onnx')
    logging.basicConfig(filename='./1.log', format='%(asctime)-15s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    # Build model
    model = MCnet_resnet18()

    base_model1 = copy.deepcopy(model)
    base_checkpoint = torch.load(checkpoint)
    base_checkpoint1 = copy.deepcopy(base_checkpoint)
    base_model1.load_state_dict(base_checkpoint1['state_dict'])
    config = './config/config_UNet_mobilenet_prune.json'
    config = json.load(open(config))
    valid_loader = get_instance(module_data, 'valid_loader', config).loader
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    criterion = getattr(module_loss, config['loss'])
    valid(base_model1, valid_loader, metrics, criterion, device='cuda')

    trained_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
    model.load_state_dict(trained_dict, strict=False)
    model.eval()
    RM_layer(copy.deepcopy(model), onnx_path)
