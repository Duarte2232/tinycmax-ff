import sys
sys.path.insert(0, '/home/duarte/Downloads/tinycmax_fixed/tinycmax_fixed')
import torch
import torch.nn as nn
from dotmap import DotMap
from tinycmax.network import FlowNetwork

net = FlowNetwork(
    encoder_channels=64,
    memory_channels=64,
    decoder_channels=64,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="zeros",
    scaling=32,
)
net.eval()
dummy = DotMap({"events": torch.zeros(1, 10, 192, 344)}, _dynamic=False)
with torch.no_grad():
    net(dummy)

print(net.decoder)