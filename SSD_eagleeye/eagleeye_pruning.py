# ------------------------------------------------------------------------------
#	Libraries
# ------------------------------------------------------------------------------
import os
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nets.ssd_loss import MultiboxLoss
from utils.anchors import get_anchors
from utils.utils import get_classes
from utils.dataloader import SSDDataset, ssd_dataset_collate
import random
import math
from nets.mobilenetv2 import ConvTranspose2d, InvertedResidual, InvertedResidual_Quantization_Friendly, ConvBNReLU1, \
    Concat1, MobileNetV2_4_2, Conv, MobileNetV2_4_2_prune, InvertedResidual_prune
from nets.ssd import SSD, SSD_Prune



def get_is_shotcut(next_block):
    block_param = copy.deepcopy(next_block)
    use_res_connect = block_param.use_res_connect
    groups = block_param.conv[0].groups
    # TODO:
    if groups > 1:
        use_res_connect = True
    return use_res_connect


def Conv2Conv_prune(block, concat_num, block_out_mask, use_res_connect, use_rate):
    features = copy.deepcopy(block.conv)
    in_mask = block_out_mask[block.from_]
    for i, m in enumerate(features):
        if isinstance(m, nn.Conv2d):
            kernel_size = features[i].kernel_size
            stride = features[i].stride
            padding = features[i].padding
            groups = features[i].groups
            out_mask = filtermask(m, use_rate)
    if use_res_connect:
        out_mask = torch.ones(len(out_mask)) > 0

    in_channels = features[0].in_channels  # [out, in, kernel, kernel]
    if in_channels != len(in_mask):
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    prune_layer = Conv(int(in_mask.sum()), int(out_mask.sum()), kernel=kernel_size[0], stride=stride[0],
                            padding=padding[0], groups=groups)
    prune_layer.index, prune_layer.from_ = block.index, block.from_
    prune_layer.conv[0].weight.data = features[0].weight.data[out_mask][:, in_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, groups]


def ConvBNReLU2ConvBNReLU_RM_prune(block, concat_num, block_out_mask, use_res_connect, use_rate):
    features = copy.deepcopy(block.conv)
    in_mask = block_out_mask[block.from_]
    for i, m in enumerate(features):
        if isinstance(m, nn.Conv2d):
            kernel_size = features[i].kernel_size
            stride = features[i].stride
            padding = features[i].padding
            groups = features[i].groups
            out_mask = filtermask(m, use_rate)

    if use_res_connect:
        out_mask = torch.ones(len(out_mask)) > 0

    in_channels = features[0].in_channels  # [out, in, kernel, kernel]
    if in_channels != len(in_mask):
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    # print('out_mask:' + str(len(out_mask)) + '   out_mask>0:' + str(out_mask.sum()))
    prune_layer = ConvBNReLU1(int(in_mask.sum()), int(out_mask.sum()), int(kernel_size[0]), stride[0], groups)
    prune_layer.index, prune_layer.from_ = block.index, block.from_
    prune_layer.conv[0].weight.data = features[0].weight.data[out_mask][:, in_mask]
    prune_layer.conv[1].weight.data = features[1].weight.data[out_mask]
    prune_layer.conv[1].bias.data = features[1].bias.data[out_mask]
    prune_layer.conv[1].running_mean = features[1].running_mean[out_mask]
    prune_layer.conv[1].running_var = features[1].running_var[out_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, groups]


def ConvTranspose2d2ConvTranspose2d_prune(block, block_out_mask):
    in_mask = block_out_mask[block.from_]
    out_mask = in_mask
    kernel_size = block.deconv.kernel_size
    stride = block.deconv.stride
    padding = block.deconv.padding
    groups = int(out_mask.sum())

    prune_layer = ConvTranspose2d(int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, 0, groups)
    prune_layer.index, prune_layer.from_ = block.index, block.from_
    # print(prune_layer.weight.data)
    prune_layer.deconv.weight.data = block.deconv.weight.data[out_mask]
    prune_layer.deconv.bias.data = block.deconv.bias.data[out_mask]
    return prune_layer, out_mask, [int(in_mask.sum()), int(out_mask.sum()), kernel_size, stride, padding, 0, groups]


def filtermask(module, use_rate):
    min_ratio, max_ratio = 0.3, 0.9
    if use_rate:
        ratio = random.uniform(min_ratio, max_ratio)
    else:
        ratio = 1
    weight = module.weight.data.abs().clone()
    weight = torch.sum(weight, dim=(1, 2, 3))
    length = weight.size()[0]
    remin_length = int(length * ratio)
    remin_length = math.ceil(remin_length / 8) * 8  # remin_length%8=0, math.ceil()向上取整
    _, index = torch.topk(weight, remin_length)
    a = weight[index[-1]]
    mask = weight >= weight[index[-1]]
    return mask


def get_InvertedResidual_rate(features, in_mask, use_rate, is_prune):
    layer_params = []  # kernel, stride, padding, groups
    out_masks = []
    is_break_off = False
    for i, m in enumerate(features):
        is_use = use_rate
        if isinstance(m, nn.Conv2d) and not isinstance(features[i+1], nn.BatchNorm2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            mask = in_mask
            groups = int(mask.sum())
            out_masks.append(mask)  # 下一层卷积操作的输入
            int_in_mask = int(in_mask.sum())
            int_mask = int(mask.sum())
            layer_params.append([int_in_mask, int_mask, kernel_size, stride, padding, groups])
            in_mask = mask
            if int_mask == 0:
                is_break_off = True

        if isinstance(m, nn.Conv2d) and isinstance(features[i+1], nn.BatchNorm2d):
            if i + 4 > len(features) and not is_prune:
                is_use = False
            kernel_size = features[i].kernel_size
            stride = features[i].stride
            padding = features[i].padding
            groups = features[i].groups
            if len(layer_params) == 0 and groups != 1:
                is_use = False

            mask = filtermask(m, is_use)
            if groups > 1:
                groups = int(mask.sum())
                if len(layer_params) > 0:
                    layer_params[-1][1] = groups
                    out_masks[-1] = mask
                    in_mask = mask
            out_masks.append(mask)  # 下一层卷积操作的输入
            int_in_mask = int(in_mask.sum())
            int_mask = int(mask.sum())
            layer_params.append([int_in_mask, int_mask, kernel_size, stride, padding, groups])
            in_mask = mask
            if int_mask == 0:
                is_break_off = True
    # print(out_masks)
    return layer_params, out_masks, is_break_off


def InvertedResidual2InvertedResidual_prune(block, concat_num, block_out_mask, next_use_res_connect, use_rate):
    block_param = copy.deepcopy(block)
    if isinstance(block, InvertedResidual) or isinstance(block, InvertedResidual_Quantization_Friendly):
        features = block_param.conv
    else:
        print('InvertedResidual2InvertedResidual_prune error ')
        return
    use_res_connect = block_param.use_res_connect
    expansion = block_param.expand_ratio

    in_mask = block_out_mask[block_param.from_]
    in_channels = features[0].in_channels  # [out, in, kernel, kernel]
    if in_channels != len(in_mask):
        in_mask = torch.cat((in_mask, block_out_mask[concat_num[-1]]), 0)
    is_prune = True if len(block_out_mask)>11 else False
    layer_params, out_masks, is_break_off = get_InvertedResidual_rate(features, in_mask, use_rate, is_prune)

    if len(in_mask) == len(out_masks[-1]):
        maks_out = out_masks[-1]
        maks = in_mask + maks_out
        out_masks[-1] = maks
        layer_params[-1][1] = int(maks.sum())

    if next_use_res_connect:
        out_masks[-1] = torch.ones(len(out_masks[-1])) > 0
        layer_params[-1][1] = int(out_masks[-1].sum())

    prune_layers = InvertedResidual_prune(layer_params, expansion, use_res_connect)
    prune_layers.index, prune_layers.from_ = block_param.index, block_param.from_
    count = 0
    for i, m in enumerate(features):
        if isinstance(m, nn.Conv2d) and not isinstance(features[i + 1], nn.BatchNorm2d):
            mask = out_masks[count]
            count += 1
            if prune_layers.conv[i].groups != 1:
                # prune_layers.conv[i - 1].weight.data = features[i - 1].weight.data[mask][in_mask]
                prune_layers.conv[i].weight.data = features[i].weight.data[mask]
            else:
                prune_layers.conv[i].weight.data = features[i].weight.data[mask][:, in_mask]
            in_mask = mask

        if isinstance(m, nn.BatchNorm2d):
            mask = out_masks[count]
            count += 1
            if prune_layers.conv[i-1].groups != 1:
                # prune_layers.conv[i - 1].weight.data = features[i - 1].weight.data[mask][in_mask]
                prune_layers.conv[i-1].weight.data = features[i-1].weight.data[mask]
            else:
                prune_layers.conv[i-1].weight.data = features[i-1].weight.data[mask][:, in_mask]
            prune_layers.conv[i].weight.data = m.weight.data[mask]
            prune_layers.conv[i].bias.data = m.bias.data[mask]
            prune_layers.conv[i].running_mean = m.running_mean[mask]
            prune_layers.conv[i].running_var = m.running_var[mask]
            in_mask = mask
    out_mask = out_masks[-1]
    block_params = []
    block_params.append(layer_params)
    block_params.append(expansion)
    block_params.append(use_res_connect)
    return prune_layers, out_mask, block_params


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
    demo_input = torch.rand(8, 3, 224, 224)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops


def valid(model, valid_data_loader, criterion, device='cuda'):
    model.to(device)
    model.eval()
    n_iter = len(valid_data_loader)
    val_loss, val_pos_loc_loss, val_pos_conf_loss, val_neg_conf_loss = 0, 0, 0, 0
    with torch.no_grad():
        # Validate
        for iteration, batch in tqdm(enumerate(valid_data_loader), total=n_iter):
            images, targets = batch[0], batch[1]
            images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
            targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()

            out = model(images)
            loss, pos_loc_loss, pos_conf_loss, neg_conf_loss = criterion.forward(targets, out)
            val_loss += loss.item()

            val_pos_loc_loss += pos_loc_loss.item()
            val_pos_conf_loss += pos_conf_loss.item()
            val_neg_conf_loss += neg_conf_loss.item()
        val_loss = val_loss / n_iter
    return val_loss


def train(model, train_data_loader, device='cuda'):
    model = model.to(device)
    model.train()
    n_iter = len(train_data_loader)
    with torch.no_grad():
        for iteration, batch in tqdm(enumerate(train_data_loader), total=n_iter):
            images, targets = batch[0], batch[1]
            images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
            out = model(images)


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def save_checkpoint(epoch, name, model, optimizer, file):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, file)


def Puring_layer(net, num_classes, train_loader, valid_loader, criterion, save_path, flops_target):
    ori_net = copy.deepcopy(net)
    ori_flops = compute_conv_flops(ori_net, cuda=True)
    while(1):
        net = copy.deepcopy(ori_net)
        net_params = []  # RM+prune后每个模块的网络参数
        features = []  # RM+prune re
        block_out_mask = []  # 每个模块输出的网络参数
        concat_num = []  # concat连接的网络层
        in_mask = torch.ones(3) > 0
        block_out_mask.append(in_mask)
        for i, block in enumerate(net.model):
            use_res_connect = False
            if i < len(net.model)-1:
                next_block = net.model[i+1]
                if isinstance(next_block, InvertedResidual):     # shotcut
                    use_res_connect = get_is_shotcut(next_block)
            if isinstance(block, Conv):
                # print('Conv')
                block_prue, in_mask, block_params = Conv2Conv_prune(block, concat_num, block_out_mask, use_res_connect, use_rate=False)
                features.append(block_prue)
            elif isinstance(block, ConvBNReLU1):
                # print('ConvBNReLU_mask')
                block_prue, in_mask, block_params = ConvBNReLU2ConvBNReLU_RM_prune(block, concat_num, block_out_mask, use_res_connect, use_rate=True)
                features.append(block_prue)
            elif isinstance(block, InvertedResidual) or isinstance(block, InvertedResidual_Quantization_Friendly):
                # print('InvertedResidual')
                block_prue, in_mask, block_params = InvertedResidual2InvertedResidual_prune(block, concat_num, block_out_mask, use_res_connect, use_rate=True)
                features.append(block_prue)
            elif isinstance(block, ConvTranspose2d):
                # print('ConvTranspose2d')
                block_prue, in_mask, block_params = ConvTranspose2d2ConvTranspose2d_prune(block, block_out_mask)
                features.append(block_prue)
            elif isinstance(block, Concat1):
                # print('Concat')
                concat_num.append(block.from_[1])
                features.append(block)
                block_params = []
                in_mask = in_mask
            else:
                features.append(block)
                block_params = []
                in_mask = []

            net_params.append(block_params)
            if i == 0:
                block_out_mask.clear()
            block_out_mask.append(in_mask)

        # load new model
        # print(net_params)
        net.model = nn.Sequential(*features)
        prune_flops = compute_conv_flops(net, cuda=True)
        ratio = prune_flops / ori_flops
        if ratio < flops_target - 0.005 or ratio > flops_target + 0.005:
            continue
        print('ori_net flops: {:f}  prune_net flops: {:f}  flops_ratio: {:f}'.format(ori_flops, prune_flops, ratio))

        val_loss = valid(net, valid_loader, criterion, device='cuda')
        print(val_loss)
        train(net, train_loader, device='cuda')
        val_loss = valid(net, valid_loader, criterion, device='cuda')
        print(val_loss)
        weight_file = save_path+'/'+str(val_loss)+'.pth'
        params_file = save_path+'/'+str(val_loss)+'.txt'
        fid = open(params_file, 'w')
        fid.write(str(net_params))   # network structure
        fid.close()

        # 重新加载模型，测试val数据集
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        save_checkpoint(
            epoch=0,
            name='prune',
            model=net,
            optimizer=optimizer,
            file=weight_file
        )

        model = SSD_Prune(net_params, block_cfg=MobileNetV2_4_2_prune, num_classes=num_classes)   # UNet_MobileNetV2_RM,UNet_MobileNetV2_4_RM
        checkpoint = torch.load(weight_file)
        model.load_state_dict(checkpoint['state_dict'])

        val_loss = valid(model, valid_loader, criterion, device='cuda')
        print(val_loss)
        from torchsummary import summary
        model = model.to("cuda")
        summary(model, (3, 256, 256))
        break


if __name__ == "__main__":
    flops_target = 0.5
    model_path = './logs/loss_2023_02_15_11_06_41_MobileNetV2_4_2_test/ep098-loss1.271-val_loss1.381.pth'
    classes_path = 'data/voc_coner_point_classes.txt'
    train_annotation_path = '2012_train_coner_point.txt'
    val_annotation_path = '2012_val_coner_point.txt'

    # 从txt中读取数据
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    train_lines = train_lines[:len(val_lines)]

    save_path, _ = os.path.split(model_path)
    input_shape = [256, 256]
    batch_size = 16
    num_workers = 0
    backbone = "MobileNetV2_4_2"
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone)
    backbone = "MobileNetV2_4_2_test"
    train_dataset = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, augment=True)
    val_dataset = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, augment=False)

    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
    criterion = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    model = SSD(backbone=backbone, num_classes=num_classes)
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    # prune_flops = compute_conv_flops(model, cuda=True)
    # from torchsummary import summary
    # model = model.to("cuda")
    # summary(model, (3, 256, 256))
    for i in range(100):
        Puring_layer(copy.deepcopy(model), num_classes, gen_train, gen_val, criterion, save_path, flops_target)



# flops_ratio = 169547840.0/282352896.0=0.6

