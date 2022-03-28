from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score_1 import Evaluator_1
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from train import parse_args

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='test', mode='test')
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)
        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            local_rank=args.local_rank,norm_layer=BatchNorm2d).to(self.device)

        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = Evaluator_1(val_dataset.NUM_CLASS)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs,  m_c,m_b = model(image)
            self.metric.add_batch(target,outputs[0])
            F1,IoU,Recall,Precision,OA = self.metric.f1_miou_test()

            F1 = F1[1]
            IoU = IoU[1]
            Recall = Recall[1]
            Precision = Precision[1]
            # Recall, Precision, mIoU, IoU,F1 = self.metric.get()
            logger.info(
                "Sample: {:d}, Validation Recall: {:.3f}, Precision: {:.3f},OA: {:.3f}, IoU: {:.3f},F1: {:.3f}".format(
                    i + 1,Recall,Precision, OA,IoU,F1))

            if self.args.save_pred:
                #seg
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(outdir + '/seg/', os.path.splitext(filename[0])[0] + '.png'))
                #body
                pred1 = torch.argmax(m_c[0], 1)
                pred1 = pred1.cpu().data.numpy()
                predict1 = pred1.squeeze(0)
                mask1 = get_color_pallete(predict1, self.args.dataset)
                mask1.save(os.path.join(outdir + '/seg_body/', os.path.splitext(filename[0])[0] + '.png'))
                #edge
                pred2 = torch.sigmoid(m_b[0])
                pred2 = pred2.cpu().data.numpy()
                pred2 = (pred2 > 0.95).astype(np.uint8)
                predict2 = pred2.squeeze(0)
                predict2 = predict2.squeeze(0)
                mask2 = get_color_pallete(predict2, self.args.dataset)
                mask2.save(os.path.join(outdir + '/seg_edge/', os.path.splitext(filename[0])[0] + '.png'))
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    CUDA_VISIBLE_DEVICES = 3

    num_gpus = 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}_test'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        a = os.path.join(outdir,'seg')
        b = os.path.join(outdir,'seg_body')
        c = os.path.join(outdir, 'seg_edge')
        if not os.path.exists(a):
            os.makedirs(a)
        if not os.path.exists(b):
            os.makedirs(b)
        if not os.path.exists(c):
            os.makedirs(c)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='test_{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
