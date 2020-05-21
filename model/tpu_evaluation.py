import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from mir_eval.separation import bss_eval_sources
import random
import config


def validate(audio, model, embedder, testloader, writer, epoch,device):
    model.eval()
    criterion = nn.MSELoss()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    with torch.no_grad():
        for i , batch in enumerate(testloader):
            dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]

            dvec_mel = dvec_mel.to(device)
            target_mag = target_mag.unsqueeze(0).to(device)
            mixed_mag = mixed_mag.unsqueeze(0).to(device)

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            est_mask = model(mixed_mag, dvec)
            est_mag = est_mask * mixed_mag

            test_loss = criterion(est_mag, target_mag).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            target_mag = target_mag[0].cpu().detach().numpy()
            est_mag = est_mag[0].cpu().detach().numpy()
            est_wav = audio.spec2wav(est_mag, mixed_phase)
            est_mask = est_mask[0].cpu().detach().numpy()

            test_sdr_avg = bss_eval_sources(target_wav, est_wav, False)[0][0]
            test_loss_avg = test_loss
            writer.log_evaluation_data(mixed_wav, target_wav, est_wav,
                                    mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                    (epoch-1),i)
            writer.log_evaluation_avg(test_loss_avg, test_sdr_avg, (epoch - 1))
            print('[xla:{}] SDR={:.2f}%'.format(xm.get_ordinal(), test_sdr_avg), flush=True)
            break
