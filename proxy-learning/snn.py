from spikingjelly.clock_driven import neuron, functional, encoding, surrogate, layer
import torch.nn as nn
import torch.nn.functional as F

class SNN(nn.Module):
    def __init__(self, T, v_threshold=2.0, v_reset=0.0):
        super().__init__()
        self.T = T
        self.vars = nn.ParameterList()
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
        )

        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64 * 4 * 5, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(64 * 4 * 5, 64 * 3 * 3, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(64 * 3 * 3, 64 * 2 * 1, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(64 * 2 * 1, 10, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        x = self.static_conv(x)
        y = self.conv(x)
        out_spikes_counter = self.fc(y)
        for t in range(1, self.T):
            if (t==0):
                out_spikes_counter = self.fc(self.conv(x))
            else:
                out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter

    def parameters(self):
        return self.vars