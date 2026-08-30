"""Audio-driven expression / blendshape (SyncAnimation Sec. 3.2)."""
import torch
import torch.nn.functional as F
from torch import nn

from .layers import FCNormRelu, ConvNormRelu
import numpy as np


class EmotionSeqEncoder(nn.Module):
    def __init__(self, bs_dim=3 * 2, noise_dim=32, time_latent_dim=16,
                 audio_dim=256, audio_layers=1, dropout=0.1, max_time=30, bs_latent_dim=256) -> None:
        super().__init__()

        leaky = True
        norm = "BN"
        self.noise_dim = noise_dim * 2
        self.audio_dim = audio_dim
        audio_hidden_nc = self.audio_dim // 2
        in_channels = time_latent_dim + bs_latent_dim + audio_hidden_nc + bs_dim
        hidden_channels = bs_latent_dim
        self.time_latent_dim = time_latent_dim

        self.max_time = max_time
        self.bs_dim = bs_dim
        self.bs_latent_dim = bs_latent_dim

        self.audio_first = nn.Sequential(
            torch.nn.Conv1d(self.audio_dim, audio_hidden_nc, kernel_size=7, padding=0, bias=True),
            nn.BatchNorm1d(audio_hidden_nc),
        )
        self.norm4 = nn.BatchNorm1d(audio_hidden_nc)
        self.audio_layers = audio_layers
        for i in range(audio_layers):
            net = nn.Sequential(nn.LeakyReLU(),
                                torch.nn.Conv1d(audio_hidden_nc, audio_hidden_nc, kernel_size=3, padding=0, dilation=3),
                                nn.BatchNorm1d(audio_hidden_nc),
                                nn.LeakyReLU(0.1)
                                )
            setattr(self, 'audio_encoder' + str(i), net)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.audio_dp = nn.Dropout(dropout)

        self.bsEmbedding = nn.Linear(self.bs_dim, self.bs_latent_dim)

        self.timeEmbedding = nn.Linear(2, self.time_latent_dim)

        self.block1 = FCNormRelu(in_features=in_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block2 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block3 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block4 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block5 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block6 = FCNormRelu(in_features=hidden_channels,
                                 out_features=hidden_channels, norm=norm, leaky=leaky)

        self.block7 = FCNormRelu(in_features=hidden_channels,
                                 out_features=self.noise_dim, norm=norm, leaky=leaky)


    def forward(self, avg_bs, bs, audio_feats, t):

        t = self.sinusoidal_time_encoding(t)
        time_enc = self.timeEmbedding(t)

        bs_ref = avg_bs
        bs_enc = self.bsEmbedding(bs)

        audio_out = self.audio_first(audio_feats)
        for i in range(self.audio_layers):
            audio_encoder = getattr(self, 'audio_encoder' + str(i))
            audio_out = audio_encoder(audio_out) + self.norm4(audio_out[:, :, 3:-3])
            audio_out = self.audio_dp(audio_out)

        audio_out = self.pooling(audio_out)
        audio_out = audio_out.view(audio_out.shape[0], -1)

        x = torch.cat([bs_ref, bs_enc, audio_out, time_enc], dim=-1)


        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        x = self.block7(x)
        mu = x[:, 0::2]
        logvar = x[:, 1::2]
        return mu, logvar

    def sinusoidal_time_encoding(self, t):
        """Sinusoidal time encoding."""
        max_time = self.max_time
        t = t % max_time
        sin_encoding = torch.sin(t / max_time * 2 * torch.pi)
        cos_encoding = torch.cos(t / max_time * 2 * torch.pi)
        return torch.cat([sin_encoding, cos_encoding], dim=1)

class EmotionSeqDecoder(nn.Module):
    def __init__(self,  bs_dim=3 * 2, noise_dim=256, time_latent_dim=16, audio_dim=256, audio_layers=1,
                 dropout=0.1, max_time=30, bs_latent_dim=256) -> None:
        super().__init__()

        leaky = True
        norm = "BN"
        self.noise_dim = noise_dim
        self.audio_dim = audio_dim
        audio_hidden_nc = self.audio_dim // 2
        in_channels = time_latent_dim + noise_dim + audio_hidden_nc + bs_dim
        hidden_channels = bs_latent_dim
        out_channels = bs_dim
        self.time_latent_dim = time_latent_dim

        self.max_time = max_time
        self.bs_dim = bs_dim
        self.bs_latent_dim = bs_latent_dim
        self.timeEmbedding = nn.Linear(2, self.time_latent_dim)

        audio_hidden_nc = self.audio_dim // 2

        self.audio_first = nn.Sequential(
            torch.nn.Conv1d(self.audio_dim, audio_hidden_nc, kernel_size=7, padding=0, bias=True),
            nn.BatchNorm1d(audio_hidden_nc),
        )
        self.norm4 = nn.BatchNorm1d(audio_hidden_nc)
        self.audio_layers = audio_layers
        for i in range(audio_layers):
            net = nn.Sequential(nn.LeakyReLU(),
                                torch.nn.Conv1d(audio_hidden_nc, audio_hidden_nc, kernel_size=3, padding=0, dilation=3),
                                nn.BatchNorm1d(audio_hidden_nc),
                                nn.LeakyReLU(0.1)
                                )
            setattr(self, 'audio_encoder' + str(i), net)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.audio_dp = nn.Dropout(dropout)

        self.block1 = FCNormRelu(in_features=in_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block2 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block3 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block4 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block5 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)
        self.block6 = FCNormRelu(in_features=hidden_channels, out_features=hidden_channels, norm=norm, leaky=leaky)

        self.output_layer = FCNormRelu(in_features=hidden_channels, out_features=out_channels, norm=norm, leaky=leaky)

    def forward(self, avg_bs, z, audio_feats, t):

        t = self.sinusoidal_time_encoding(t)
        time_enc = self.timeEmbedding(t)

        bs_ref = avg_bs

        audio_out = self.audio_first(audio_feats)
        for i in range(self.audio_layers):
            audio_encoder = getattr(self, 'audio_encoder' + str(i))
            audio_out = audio_encoder(audio_out) + self.norm4(audio_out[:, :, 3:-3])
            audio_out = self.audio_dp(audio_out)

        audio_out = self.pooling(audio_out)
        audio_out = audio_out.view(audio_out.shape[0], -1)

        x = torch.cat([bs_ref, z, audio_out, time_enc], dim=-1)


        x = self.block1(x)


        x = self.block2(x)


        x = self.block3(x)


        x = self.block4(x)


        x = self.block5(x)


        x = self.block6(x)

        x = self.output_layer(x)

        return x
    def sinusoidal_time_encoding(self, t):
        """Sinusoidal time encoding."""
        max_time = self.max_time
        t = t % max_time
        sin_encoding = torch.sin(t / max_time * 2 * torch.pi)
        cos_encoding = torch.cos(t / max_time * 2 * torch.pi)
        return torch.cat([sin_encoding, cos_encoding], dim=1)





class EmotionCVAE(nn.Module):
    def __init__(self, audio_dim=256, bs_dim=6, noise_dim=256, bs_latent_dim=256) -> None:
        super().__init__()


        self.encoder = EmotionSeqEncoder(audio_dim=audio_dim, bs_dim=bs_dim, bs_latent_dim=bs_latent_dim,
                                    audio_layers=2, dropout=0.1, max_time=10000, noise_dim=noise_dim)


        self.decoder = EmotionSeqDecoder(audio_dim=audio_dim, bs_dim=bs_dim, bs_latent_dim=bs_latent_dim,
                                    audio_layers=2, dropout=0.1, max_time=10000, noise_dim=noise_dim)


    def forward(self, avg_bs, bs, audio_feats, t, external_code=None):

        if external_code is not None:
            x_recon = self.decoder(avg_bs=avg_bs, z=external_code,
                                   audio_feats=audio_feats, t=t)

            return x_recon, 0, 1

        mu, logvar = self.encoder(avg_bs=avg_bs, bs=bs, audio_feats=audio_feats, t=t)

        eps = torch.randn(logvar.shape, device=logvar.device)
        code = mu + torch.exp(0.5 * logvar) * eps

        x_recon = self.decoder(avg_bs=avg_bs, z=code,
                               audio_feats=audio_feats, t=t)


        return x_recon, mu, logvar


class EmotionUNet1D(nn.Module):
    def __init__(self, audio_dim=256, bs_dim=13, hidden_dim=256):
        super().__init__()

        leaky = True
        norm = "BN"
        in_features = audio_dim + bs_dim + 256 + 128

        self.e0 = FCNormRelu(in_features=in_features, out_features=hidden_dim, norm=norm, leaky=leaky)
        self.e1 = FCNormRelu(in_features=hidden_dim, out_features=hidden_dim // 2, norm=norm, leaky=leaky)
        self.e2 = FCNormRelu(in_features=hidden_dim // 2, out_features=hidden_dim // 4, norm=norm, leaky=leaky)
        self.e3 = FCNormRelu(in_features=hidden_dim // 4, out_features=hidden_dim // 8, norm=norm, leaky=leaky)
        self.e4 = FCNormRelu(in_features=hidden_dim // 8, out_features=hidden_dim // 16, norm=norm, leaky=leaky)
        self.e5 = FCNormRelu(in_features=hidden_dim // 16, out_features=hidden_dim // 32, norm=norm, leaky=leaky)
        self.e6 = FCNormRelu(in_features=hidden_dim // 32, out_features=hidden_dim // 64, norm=norm, leaky=leaky)

        self.d5 = FCNormRelu(in_features=hidden_dim // 32, out_features=hidden_dim // 16, norm=norm, leaky=leaky)
        self.d4 = FCNormRelu(in_features=hidden_dim // 16, out_features=hidden_dim // 8, norm=norm, leaky=leaky)
        self.d3 = FCNormRelu(in_features=hidden_dim // 8, out_features=hidden_dim // 4, norm=norm, leaky=leaky)
        self.d2 = FCNormRelu(in_features=hidden_dim // 4, out_features=hidden_dim // 2, norm=norm, leaky=leaky)
        self.d1 = FCNormRelu(in_features=hidden_dim // 2, out_features=hidden_dim, norm=norm, leaky=leaky)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d5 = self.d5(F.interpolate(e6.unsqueeze(1), size=e5.size(-1), mode='linear').squeeze(1) + e5)
        d4 = self.d4(F.interpolate(d5.unsqueeze(1), size=e4.size(-1), mode='linear').squeeze(1) + e4)
        d3 = self.d3(F.interpolate(d4.unsqueeze(1), size=e3.size(-1), mode='linear').squeeze(1) + e3)
        d2 = self.d2(F.interpolate(d3.unsqueeze(1), size=e2.size(-1), mode='linear').squeeze(1) + e2)
        d1 = self.d1(F.interpolate(d2.unsqueeze(1), size=e1.size(-1), mode='linear').squeeze(1) + e1)

        return d1

class EmotionAttrMapNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, norm="BN", leaky=True):
        super(EmotionAttrMapNet, self).__init__()


        self.arkit_fc1 = FCNormRelu(in_features=in_channels,
                                                 out_features=hidden_channels,
                                                 norm=norm,
                                                 leaky=leaky)
        self.arkit_fc2 = FCNormRelu(in_features=in_channels,
                                                 out_features=hidden_channels,
                                                 norm=norm,
                                                 leaky=leaky)


        self.arkit_output_1 = nn.Linear(hidden_channels, 5)
        self.arkit_output_2 = nn.Linear(hidden_channels, 2)

    def forward(self, arkit_output):
        arkit_feat1 = self.arkit_fc1(arkit_output)
        arkit_feat2 = self.arkit_fc2(arkit_output)

        arkit_1 = self.arkit_output_1(arkit_feat1)
        arkit_2 = self.arkit_output_2(arkit_feat2)
        arkit = torch.cat((arkit_1, arkit_2), dim=1)
        return arkit

class AudioEmotionSyncer(nn.Module):
    def __init__(self, audio_dim=256, audio_layers=3, bs_dim=7, hidden_dim=256, dropout=0.1):
        super(AudioEmotionSyncer, self).__init__()

        self.autoencoder = EmotionCVAE(audio_dim=256, bs_dim=bs_dim, noise_dim=256, bs_latent_dim=64)
        self.unet_bs = EmotionUNet1D(audio_dim=audio_dim // 2, bs_dim=bs_dim, hidden_dim=hidden_dim)

        self.audio_dim = audio_dim
        audio_hidden_nc = audio_dim // 2

        self.audio_first = nn.Sequential(
            torch.nn.Conv1d(self.audio_dim, audio_hidden_nc, kernel_size=7, padding=0, bias=True),
            nn.BatchNorm1d(audio_hidden_nc),
        )
        self.norm4 = nn.BatchNorm1d(audio_hidden_nc)
        self.audio_layers = audio_layers
        for i in range(audio_layers):
            net = nn.Sequential(nn.LeakyReLU(),
                                torch.nn.Conv1d(audio_hidden_nc, audio_hidden_nc, kernel_size=3, padding=0, dilation=3),
                                nn.BatchNorm1d(audio_hidden_nc),
                                nn.LeakyReLU(0.1)
                                )
            setattr(self, 'audio_encoder' + str(i), net)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.bs_mapp = FCNormRelu(in_features=5,
                                     out_features=128,
                                     norm="BN",
                                     leaky=True,
                                     drop_out=0.60)

        self.blink_mapp = FCNormRelu(in_features=2,
                                     out_features=256,
                                     norm="BN",
                                     leaky=True,
                                     drop_out=0.50)

        self.attr_map_net = EmotionAttrMapNet(in_channels=hidden_dim, hidden_channels=hidden_dim // 2)

    def forward(self, audio, bs=None, t=None, external_code=None):
        """Audio + blendshape -> ARKit coeffs, VAE mu/logvar."""
        avg_bs, bs_data = bs[0], bs[1]

        arkit, mu_pred, logvar_pred = self.autoencoder(avg_bs=avg_bs, bs=bs_data, audio_feats=audio,
                                                       t=t, external_code=external_code)

        audio_out = self.audio_first(audio)
        for i in range(self.audio_layers):
            audio_encoder = getattr(self, 'audio_encoder' + str(i))
            audio_out = audio_encoder(audio_out) +  self.norm4(audio_out[:, :, 3:-3])
        audio_out = self.pooling(audio_out)
        audio_out = audio_out.view(audio_out.shape[0], -1)

        bs_expanded = arkit.expand(audio.size(0), -1)
        bs_ = bs_data[:, :7].expand(audio.size(0), -1)

        bs_diff_true_map = self.bs_mapp(bs_[:, :5])
        blink_diff_true_map = self.blink_mapp(bs_[:, 5:7])
        bs_diff_true_expanded = torch.cat([bs_diff_true_map, blink_diff_true_map], dim=1)


        bs_input = torch.cat([audio_out, bs_expanded, bs_diff_true_expanded], dim=1)
        bs_output = self.unet_bs(bs_input)

        arkit = self.attr_map_net(bs_output)

        return arkit, mu_pred, logvar_pred




if __name__ == '__main__':
    audio = torch.randn(3, 256, 27)
    bs = torch.randn(3, 6)
    avg_bs = torch.randn(1, 6)
    t = torch.randn(3, 1)
    model = EmotionCVAE()

    y = model(avg_bs=avg_bs, bs=bs, audio_feats=audio, t=t)

