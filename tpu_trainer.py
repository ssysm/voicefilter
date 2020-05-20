import argparse
import logging
import os
import time
import torch
import config
from torch import nn
from torch.optim.lr_scheduler import StepLR

import torch_xla
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm

from utils.writer import MyWriter
from utils.dataloader import create_dataloader
from utils.audio import Audio
from utils.model_saver import model_saver
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from model.train import train
from model.evaluation import validate


def trainer(model_name):
    chkpt_path = None #@param
    device = xm.xla_device()
    pt_dir = os.path.join('.', config.log['chkpt_dir'], model_name)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join('.', config.log['log_dir'], model_name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (model_name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = MyWriter(log_dir)

    trainloader = create_dataloader(train=True)
    testloader = create_dataloader(train=False)

    embedder_pt = torch.load('/dirve/My Drive/ColabDisk/embedder.pt')
    embedder = SpeechEmbedder().to(device)
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    model = VoiceFilter().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.train['adam'])
    audio = Audio()
    
    starting_epoch = 1

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint_file = torch.load(chkpt_path)
        model.load_state_dict(checkpoint_file['model'])
        optimizer.load_state_dict(checkpoint_file['optimizer'])
        starting_epoch = checkpoint_file['epoch']
    else:
        logger.info("Starting new training run")

    for epoch in range(starting_epoch, config.train['epoch'] + 1):
        para_loader  =  pl.ParallelLoader(trainloader, [device])
        train(para_loader.per_device_loader(device),embedder,model,optimizer,writer,logger,epoch,pt_dir)
        xm.master_print("Finished training epoch {}".format(epoch))
        logger.info("Starting to validate epoch...")
        para_loader  =  pl.ParallelLoader(testloader, [device])
        validate(audio,model,embedder,para_loader,writer,epoch)

    model_saver(model,optimizer,pt_dir,config.train['epoch'])