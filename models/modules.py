import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import Swish

class ConvDownsampling(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super(ConvDownsampling, self).__init__()
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.pw_conv1 = nn.Conv1d(d_in, 2 * d_out, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(d_out, d_out, kernel_size=3, stride=2, padding=1, groups=d_out)
        self.batch_norm = nn.BatchNorm1d(d_out)
        self.swish = Swish()
        self.pw_conv2 = nn.Conv1d(d_out, d_out, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(d_in, d_out, kernel_size=1)  # For residual

    def forward(self, x):
        # (batch, seq_len, d_in)
        x_ln = self.layer_norm(x)
        x_ln = x_ln.transpose(1, 2)  

        out = self.pw_conv1(x_ln)
        out = self.glu(out)
        out = self.dw_conv(out)
        out = self.batch_norm(out)
        out = self.swish(out)
        out = self.pw_conv2(out)
        x_dropout = self.dropout(out)

        x_res = self.proj(x_ln)
        print(f"x_res shape: {x_res.shape}, x_dropout shape: {x_dropout.shape}")
        
        if x_res.shape[2] != x_dropout.shape[2]:
            x_res = F.max_pool1d(x_res, kernel_size=3, stride=2, padding=1)

        out = x_dropout + x_res
        out = out.transpose(1, 2)  # (batch, seq_len, d_out)

        return out


if __name__ == "__main__":
    # Example usage
    model = ConvDownsampling(d_in=128, d_out=64)
    x = torch.randn(32, 100, 128)  # (batch_size, seq_len, d_in)
    output = model(x)
    print(output.shape)  # Should be (32, 50, 64) after downsampling