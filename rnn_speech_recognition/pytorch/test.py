import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch_ccl

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)


if __name__ == "__main__":
    
    os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
  
    # Initialize the process group with ccl backend
    dist.init_process_group(backend='ccl')

    model = Model()
    if dist.get_world_size() > 1:
        print('ddp')
        model=DDP(model)

    for i in range(3):
        input = torch.randn(2, 4)
        labels = torch.randn(2, 5)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # forward
        res = model(input)
        print(res)
        L=loss_fn(res, labels)

        # backward
        L.backward()

        # update
        optimizer.step()