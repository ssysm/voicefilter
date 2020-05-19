import os
import torch

def model_saver(model,path, epoch, step):
    save_path = os.path.join(path, 'VoiceFilter-E%d-S%d.pt' % (epoch,step))
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step
    },save_path)