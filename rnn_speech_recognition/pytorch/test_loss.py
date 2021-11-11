import torch
from warprnnt_pytorch import RNNTLoss
rnnt_loss = RNNTLoss()
acts = torch.FloatTensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.6, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.8, 0.1]],
                            [[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.1, 0.1],
                            [0.7, 0.1, 0.2, 0.1, 0.1]]]])
labels = torch.IntTensor([[1, 1]])
act_length = torch.IntTensor([2])
label_length = torch.IntTensor([2])

#acts = torch.ones(128,3,3,5)
#labels = torch.ones(128,2)
#act_length = torch.ones(128) * 3
#label_length = torch.ones(128) * 2
# fail with 128 bs
acts = torch.ones(64, 264, 94, 1024)
labels = torch.ones(64, 93)
act_length = torch.ones(64) * 264
label_length = torch.ones(64) * 93

if acts.dtype != torch.float:
    acts = acts.float()
if labels.dtype != torch.int32:
    labels = labels.int()
if act_length.dtype != torch.int32:
    act_length = act_length.int()
if label_length.dtype != torch.int32:
    label_length = label_length.int()

print(acts.shape)
print(labels.shape)
print(act_length.shape)
print(label_length.shape)

acts = torch.autograd.Variable(acts, requires_grad=True)
labels = torch.autograd.Variable(labels)
act_length = torch.autograd.Variable(act_length)
label_length = torch.autograd.Variable(label_length)
loss = rnnt_loss(acts, labels, act_length, label_length)
print('after loss')
loss.backward()