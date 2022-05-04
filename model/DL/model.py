from torch import nn

from model.DL.atten import AttenLayer


class Model(nn.Module):
    def __init__(self, d_input):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(1, 512)
        self.atten1 = AttenLayer(d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8, dropout=0.1)
        self.atten2 = AttenLayer(d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8, dropout=0.1)
        self.atten3 = AttenLayer(d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8, dropout=0.1)
        self.layer2 = nn.Linear(512 * d_input, 2)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.layer1(x)
        x = self.atten1(x, x, x)
        x = self.atten2(x, x, x)
        x = self.atten3(x, x, x)
        x = self.layer2(x.reshape(x.size(0), -1))
        return x
