"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


__all__ = ['get_segmentation_loss', 'mixloss_3']


def edge_canny(inp):
    x_size = inp.size()
    im_arr = inp.cpu().numpy().astype(np.uint8)
    canny = np.zeros((x_size[0], x_size[1], x_size[2]))
    for i in range(x_size[0]):
        canny[i] = cv2.Canny(im_arr[i], 10, 100, apertureSize=7)
    canny = torch.from_numpy(canny).cuda().long()
    return canny
def body_canny(inp):
    x_size = inp.size()
    im_arr = inp.cpu().numpy().astype(np.uint8)
    canny = np.zeros((x_size[0], x_size[1], x_size[2]))
    for i in range(x_size[0]):
        canny[i] = cv2.Canny(im_arr[i], 10, 100)
    canny = torch.from_numpy(canny).cuda().long()
    body = inp - canny
    return body

def tensor_erode(bin_img,ksize = 7):
    B,H,W = bin_img.shape
    pad = (ksize - 1) // 2
    fbin_img = F.pad(bin_img,[pad,pad,pad,pad],mode='constant', value=0)
    patches = fbin_img.unfold(dimension = 1,size = ksize, step = 1)
    patches = patches.unfold(dimension = 2, size=ksize, step=1)
    eroded, _ = patches.reshape(B,H,W,-1).min(dim = -1)
    boundary = bin_img - eroded
    im_arr = boundary.cpu().numpy().astype(np.uint8)
    return boundary

def tensor_center(bin_img,ksize = 7):
    B,H,W = bin_img.shape
    pad = (ksize - 1) // 2
    fbin_img = F.pad(bin_img,[pad,pad,pad,pad],mode='constant', value=0)
    patches = fbin_img.unfold(dimension = 1,size = ksize, step = 1)
    patches = patches.unfold(dimension = 2, size=ksize, step=1)
    eroded, _ = patches.reshape(B,H,W,-1).min(dim = -1)
    im_arr = eroded.cpu().numpy().astype(np.uint8)
    return eroded

class mixloss_3(nn.Module):
    def __init__(self,aux_weight=20,**kwargs):
        super(mixloss_3, self).__init__()
        self.crossloss = nn.CrossEntropyLoss()
        self.edgeloss = bce()
        self.aux_weight = aux_weight

    def forward(self,output,m_c,m_b,target,**kwargs):
        target_b = tensor_erode(target,ksize=7)
        target_c = tensor_center(target,ksize=3)
        output = output[0]
        m_c = m_c[0]
        m_b = m_b[0]


        loss1 = self.crossloss(output,target)
        loss2 = self.crossloss(m_c,target_c)
        loss3 = self.edgeloss(m_b,target_b)

        loss_seg = loss1
        loss_body = loss2
        loss_edge = loss3
        loss_sum = loss_seg + loss_body + self.aux_weight*loss_edge
        return dict(loss= loss_sum),dict(loss_seg = loss_seg),dict(loss_edge = loss_edge)

class bce(nn.Module):
    def __init__(self):
        super(bce, self).__init__()

    def forward(self, input, target):
        n, c, h, w = input.size()
        target = target.unsqueeze(dim=1)

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight).cuda()
        log_p = log_p.cuda()
        target_t = target_t.cuda().float()

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)

        return loss



def get_segmentation_loss(aux_weight=1, **kwargs):
    return mixloss_3(aux_weight,**kwargs)

