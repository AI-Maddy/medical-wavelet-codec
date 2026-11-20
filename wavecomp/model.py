"""Neural entropy model: modality-aware context encoder + ConvLSTM across subbands.

The subbands of the integer wavelet pyramid are coded as a sequence, coarse to
fine:  LL_L, HL_L, LH_L, HH_L, HL_{L-1}, ... , HH_1.  For each subband the model
predicts, for every coefficient, the mean and scale of a discretised Laplace
distribution, which the range coder then uses.

Context available for a subband at level l (all of it is known to the decoder):
  * the low band LL_l at the same resolution (reconstructed from coarser levels),
  * the sibling subbands of level l that were already coded,
  * the three parent subbands of level l+1, upsampled x2,
  * the ConvLSTM hidden state, which carries information along the subband
    sequence (the "sequential dependencies"); it is upsampled x2 at each new level,
  * a one-hot band type.
A squeeze-and-excitation gate conditioned on a learned modality embedding
re-weights the context channels ("modality-aware channel attention").

Inside each subband, coefficients are coded in 2x2 polyphase order: phase (0,0)
first (recursively, so it can use its own coarser grid), then (1,1), (0,1),
(1,0), each conditioned on the phases already coded ("causal context windows").
Every phase is coded in parallel, so decoding needs a fixed, small number of
network evaluations per subband rather than one per coefficient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BAND_TYPES = 4      # LL, HL, LH, HH
N_PHASES = 4
MODALITIES = ["xray", "ct", "mri"]


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k=3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class EntropyModel(nn.Module):
    def __init__(self, ch=32, head_ch=24, hid=16, use_lstm=True, use_attention=True,
                 n_mod=len(MODALITIES), mixture=0):
        """mixture = 0: discretised Laplace (2 outputs per coefficient).
        mixture = K > 0: discretised mixture of K logistics (DMOL, 3K outputs:
        mixture logits, means, log-scales)."""
        super().__init__()
        self.cfg = dict(ch=ch, head_ch=head_ch, hid=hid, use_lstm=use_lstm,
                        use_attention=use_attention, n_mod=n_mod, mixture=mixture)
        self.mixture = mixture
        self.hid = hid
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        cin = 1 + 3 + 3 + hid + N_BAND_TYPES
        self.ctx = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
        self.mod_emb = nn.Embedding(n_mod, ch)
        self.se = nn.Sequential(nn.Linear(ch, ch // 2), nn.GELU(), nn.Linear(ch // 2, ch))
        if use_lstm:
            self.lstm = ConvLSTMCell(ch + 1, hid)
        hin = ch + 2 + N_PHASES
        self.head = nn.Sequential(
            nn.Conv2d(hin, head_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(head_ch, head_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(head_ch, 3 * mixture if mixture else 2, 1))
        if mixture:
            # spread the initial component means so the mixture starts multi-modal
            with torch.no_grad():
                b = self.head[-1].bias
                b.zero_()
                b[mixture:2 * mixture] = torch.linspace(-1, 1, mixture)

    # ---- context ---------------------------------------------------------
    def context(self, low, sibs, parents, h, band, mod):
        B, _, H, W = sibs.shape
        onehot = torch.zeros(B, N_BAND_TYPES, H, W, dtype=sibs.dtype)
        onehot[:, band] = 1
        f = self.ctx(torch.cat([low, sibs, parents, h, onehot], 1))
        if self.use_attention:
            z = f.mean((2, 3)) + self.mod_emb(mod)
            f = f * torch.sigmoid(self.se(z))[:, :, None, None]
        else:
            f = f + self.mod_emb(mod)[:, :, None, None]
        return f

    def params(self, feat, known, mask, phase):
        B, _, H, W = feat.shape
        ph = torch.zeros(B, N_PHASES, H, W, dtype=feat.dtype)
        ph[:, phase] = 1
        out = self.head(torch.cat([feat, known, mask, ph], 1))
        if self.mixture:
            K = self.mixture
            return out[:, :K], out[:, K:2 * K], out[:, 2 * K:].clamp(-6, 8)
        mu, logb = out[:, :1], out[:, 1:]
        return mu, logb.clamp(-6, 8)

    def step(self, feat, xn, h, c):
        if not self.use_lstm:
            return h, c
        return self.lstm(torch.cat([feat, xn], 1), h, c)

    def zero_state(self, B, H, W):
        z = torch.zeros(B, self.hid, H, W)
        return z, z.clone()


def upsample(x, shape):
    return F.interpolate(x, size=shape, mode="nearest")
