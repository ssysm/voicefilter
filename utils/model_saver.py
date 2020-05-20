import os
import torch

def model_saver(model, optimizer,path, epoch, step):
    save_path = os.path.join(path, 'VoiceFilter.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    },save_path)