import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import time
import traceback

import config
from utils.model_saver import model_saver

def train(embedder, model, optimizer, trainloader, writer, logger, epoch, pt_dir,device):
    try:
        tracker = xm.RateTracker()
        criterion = nn.MSELoss()
        model.train()
        step = 0
        for batch_idx, (dvec_mel, target_mag, mixed_mag) in enumerate(trainloader):
            target_mag, mixed_mag = target_mag.to(device), mixed_mag.to(device)

            dvec_list = list()
            for mel in dvec_mel:
                mel = mel.to(device)
                dvec = embedder(mel)
                dvec_list.append(dvec)
            dvec = torch.stack(dvec_list, dim=0)
            dvec = dvec.detach()
            #mask model
            optimizer.zero_grad()
            mask = model(mixed_mag, dvec)
            output = mixed_mag * mask
            #calculate loss, the paper says it use powerlaw, but we don't do it here
            loss = criterion(output, target_mag)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(len(output))
            loss = loss.item()
            #log
            step += len(output)
            logger.info('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
            xm.get_ordinal(), batch_idx, loss, tracker.rate(),
            tracker.global_rate(), time.asctime()))
            if step % config.train['ckpt_interval'] == 0 :
                model_saver(model,optimizer,pt_dir,epoch)
                logger.info("Saved Checkpoint at Epoch%d,Step%d" % (epoch, step))
            
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
