import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import Swish
import torchaudio

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

class SpecAugment(nn.Module):

    """Spectrogram Augmentation

    Args:
        spec_augment: whether to apply spec augment
        mF: number of frequency masks
        F: maximum frequency mask size
        mT: number of time masks
        pS: adaptive maximum time mask size in %

    References:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition, Park et al.
        https://arxiv.org/abs/1904.08779

        SpecAugment on Large Scale Datasets, Park et al.
        https://arxiv.org/abs/1912.05533

    """

    def __init__(self, spec_augment, mF, F, mT, pS):
        super(SpecAugment, self).__init__()
        self.spec_augment = spec_augment
        self.mF = mF
        self.F = F
        self.mT = mT
        self.pS = pS

    def forward(self, x, x_len):
        # x: (batch, freq, time)
        # Spec Augment
        if self.spec_augment:
        
            # Frequency Masking
            for _ in range(self.mF):
                x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(x)

            # Time Masking
            for b in range(x.size(0)):
                T = int(self.pS * x_len[b])
                for _ in range(self.mT):
                    x[b:b+1, :, :x_len[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(x[b:b+1, :, :x_len[b]])

        return x

# if __name__ == "__main__":
#     # Example usage
#     batch_size, seq_len, d_in = 32, 100, 128
#     x = torch.randn(batch_size, seq_len, d_in)  # (batch_size, seq_len, d_in)
    
#     # Test ConvDownsampling
#     model = ConvDownsampling(d_in=d_in, d_out=64)
#     output = model(x)
#     print("ConvDownsampling output shape:", output.shape)  # Should be (32, 50, 64) after downsampling
    
#     # Giả sử đầu vào cho SpecAugment là (batch, freq, time)
#     x_spec = torch.randn(batch_size, 80, 120)  # ví dụ đặc trưng spectrogram
#     x_len = torch.full((batch_size,), 120, dtype=torch.long)  # độ dài thực tế cho mỗi batch
#     spec_aug = SpecAugment(spec_augment=True, mF=2, F=10, mT=2, pS=0.2)
#     x_aug = spec_aug(x_spec, x_len)
#     print("SpecAugment output shape:", x_aug.shape)