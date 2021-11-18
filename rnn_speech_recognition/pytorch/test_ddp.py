from torch.nn.parallel import DistributedDataParallel as DDP
import distributed as dist
import torch
import torch.nn as nn
import intel_pytorch_extension as ipex
import torch_ccl
import os

from rnnt.model import RNNT

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)

input = torch.randn(2, 4).to(ipex.DEVICE)


dist.init_distributed(backend='ccl')

# model = RNNT(n_classes=26, in_feats=240, enc_n_hid=1024, enc_pre_rnn_layers=2, enc_post_rnn_layers=3, enc_stack_time_factor=2, enc_dropout=0.1, 
# pred_dropout=0.3, joint_dropout=0.3, pred_n_hid=512, pred_rnn_layers=2, joint_n_hid=512, forget_gate_bias=1.0)
model = Model()
print(model)

model = DDP(model)
# print(model(input))