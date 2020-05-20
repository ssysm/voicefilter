# Voice Filter 

Unofficial&Modified PyTorch implementation of Google AI's:
[VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826).

This project was inspired by and based on: [mindslab-ai/voicefilter](https://github.com/mindslab-ai/voicefilter)

Modifcations:

 - This network added two `Dropout2d` layer before `LSTM` hidden layer and before `FC 2` Layer.

 - Multi-GPU Training Support


## Model Info
```bash
VoiceFilter(
  (conv): Sequential(
    (0): ZeroPad2d(padding=(3, 3, 0, 0), value=0.0)
    (1): Conv2d(1, 64, kernel_size=(1, 7), stride=(1, 1))
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): ZeroPad2d(padding=(0, 0, 3, 3), value=0.0)
    (5): Conv2d(64, 64, kernel_size=(7, 1), stride=(1, 1))
    (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): ZeroPad2d(padding=(2, 2, 2, 2), value=0.0)
    (9): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
    (12): ZeroPad2d(padding=(2, 2, 4, 4), value=0.0)
    (13): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), dilation=(2, 1))
    (14): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): ReLU()
    (16): ZeroPad2d(padding=(2, 2, 8, 8), value=0.0)
    (17): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), dilation=(4, 1))
    (18): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU()
    (20): ZeroPad2d(padding=(2, 2, 16, 16), value=0.0)
    (21): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), dilation=(8, 1))
    (22): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (23): ReLU()
    (24): ZeroPad2d(padding=(2, 2, 32, 32), value=0.0)
    (25): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), dilation=(16, 1))
    (26): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (27): ReLU()
    (28): Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
    (29): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (30): ReLU()
  )
  (dropout1): Dropout2d(p=0.25)
  (lstm): LSTM(5064, 400, batch_first=True, bidirectional=True)
  (fc1): Linear(in_features=800, out_features=600, bias=True)
  (dropout2): Dropout2d(p=0.5)
  (fc2): Linear(in_features=600, out_features=601, bias=True)
)
```

## Todo:
 - Impl Power-Law Compression as described in paper.
 - Adapt Quick Start Guide
 - Refact `generator.py`, but the original one will work.
 - Distributed Training