import torch
from tqdm import tqdm
import math
from utils.utils import get_lr
import numpy as np
from nets.ssd_loss import updateBN


def fit_one_epoch(model_train, model, ssd_loss, bbox_util, loss_history, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, num_batch=0, num_warmup=None, END_EPOCH=0, sparsity=0):
    train_pos_loc_loss, train_pos_conf_loss, train_neg_conf_loss = 0, 0, 0
    val_pos_loc_loss, val_pos_conf_loss, val_neg_conf_loss = 0, 0, 0
    train_loss = 0
    val_loss = 0

    model_train.train()
    # print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if num_batch > 0:   # use cosine lr
                num_iter = iteration + num_batch * (epoch - 1)

                if num_iter < num_warmup:
                    # warm up
                    lf = lambda x: ((1 + math.cos(x * math.pi / END_EPOCH)) / 2) * (1 - 0.2) + 0.2  # cosine
                    xi = [0, num_warmup]
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(num_iter, xi, [0.8, 0.937])

            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor) 
            #----------------------#
            #   前向传播
            #----------------------#
            out = model_train(images)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   计算损失
            #----------------------#
            loss, pos_loc_loss, pos_conf_loss, neg_conf_loss = ssd_loss.forward(targets, out)
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            if sparsity > 0:
                updateBN(model_train, sparsity)

            optimizer.step()

            train_loss += loss.item()

            train_pos_loc_loss += pos_loc_loss.item()
            train_pos_conf_loss += pos_conf_loss.item()
            train_neg_conf_loss += neg_conf_loss.item()

            pbar.set_postfix(**{'train_loss'    : train_loss / (iteration + 1),
                                'train_pos_loc_loss': train_pos_loc_loss / (iteration + 1),
                                'train_pos_conf_loss': train_pos_conf_loss / (iteration + 1),
                                'train_neg_conf_loss': train_neg_conf_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
                
    # print('Finish Train')

    model_train.eval()
    # print('Start Validation')
    TP, PREDS, P_N = {}, {}, {}
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = torch.from_numpy(targets).type(torch.FloatTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = torch.from_numpy(targets).type(torch.FloatTensor) 

                out = model_train(images)
                optimizer.zero_grad()
                loss, pos_loc_loss, pos_conf_loss, neg_conf_loss = ssd_loss.forward(targets, out)
                val_loss += loss.item()

                val_pos_loc_loss += pos_loc_loss.item()
                val_pos_conf_loss += pos_conf_loss.item()
                val_neg_conf_loss += neg_conf_loss.item()

                pbar.set_postfix(**{'val_loss'    : val_loss / (iteration + 1),
                                    'val_pos_loc_loss': val_pos_loc_loss / (iteration + 1),
                                    'val_pos_conf_loss': val_pos_conf_loss / (iteration + 1),
                                    'val_neg_conf_loss': val_neg_conf_loss / (iteration + 1),
                                    'lr'            : get_lr(optimizer)})
                pbar.update(1)

                results = bbox_util.decode_box(out)
                if len(results[0]) != 0:
                    pass
                TP, PREDS, P_N = bbox_util.get_ap(results, batch[2], TP, PREDS, P_N)

    bbox_util.print_recall_precision(TP, PREDS, P_N)
    # print('Finish Validation')

    loss_history.append_loss(train_loss/epoch_step, val_loss/epoch_step_val, epoch+1, get_lr(optimizer))
    print('Epoch:' + str(epoch+1) + '/' + str(Epoch))

    print('Train Loss: %.3f Train pos_loc_loss: %.3f Train pos_conf_loss: %.3f Train neg_conf_loss: %.3f || '
          'Val Loss: %.3f ' 'Val pos_loc_loss: %.3f ' 'Val pos_conf_loss: %.3f ' 'Val neg_conf_loss: %.3f ' % (
        train_loss / epoch_step, train_pos_loc_loss / epoch_step, train_pos_conf_loss / epoch_step, train_neg_conf_loss / epoch_step,
        val_loss / epoch_step_val, val_pos_loc_loss / epoch_step_val, val_pos_conf_loss / epoch_step_val, val_neg_conf_loss / epoch_step_val))

    torch.save(model.state_dict(), loss_history.save_path+'/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_loss / epoch_step, val_loss / epoch_step_val))
