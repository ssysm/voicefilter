import argparse
import logging
import os
import time
import torch
import config
from torch import nn
from torch.optim.lr_scheduler import StepLR

from utils.writer import MyWriter
from utils.dataloader import create_dataloader
from utils.audio import Audio
from utils.model_saver import model_saver
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from model.train import train
from model.evaluation import validate


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Voice Filter')
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to last checkpoint')
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model. Used for both logging and saving checkpoints.")
    args = parser.parse_args()

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    pt_dir = os.path.join(args.base_dir, config.log['chkpt_dir'], args.model)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(args.base_dir, config.log['log_dir'], args.model)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = MyWriter(log_dir)

    trainloader = create_dataloader(train=True)
    testloader = create_dataloader(train=False)

    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder().cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    model = nn.DataParallel(VoiceFilter())
    optimizer = torch.optim.Adam(model.parameters(),lr=config.train['adam'])
    audio = Audio()
    
    starting_step = 0
    starting_epoch = 1

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint_file = torch.load(chkpt_path)
        model.load_state_dict(checkpoint_file['model'])
        starting_epoch = checkpoint_file['epoch']
        starting_step = checkpoint_file['step']
    else:
        logger.info("Starting new training run")

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(starting_epoch, config.train['epoch'] + 1):
        train(embedder,model,optimizer,trainloader,writer,logger,epoch,pt_dir,starting_step)
        validate(audio,model,embedder,testloader,writer,epoch)
        scheduler.step()
        starting_step = 0

    model_saver(model,pt_dir,config.train['epoch'],config.train['train_step_pre_epoch'])

if __name__ == '__main__':
    main()