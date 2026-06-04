import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import math


class SpatioTemporalEncoder(nn.Module):
    
    def __init__(self, d_model, in_ch=1):
        super().__init__()
      
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
           
            nn.AdaptiveAvgPool3d((None, 1, 1))  
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x):
      
        b, t, c, h, w = x.shape
        feat = self.cnn3d(x.permute(0, 2, 1, 3, 4))  
        feat = feat.squeeze(-1).squeeze(-1).permute(0, 2, 1)  


class Model(nn.Module):
    

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.use_visual = True 
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
       
        img_ch = getattr(configs, 'img_channel', 1)
        self.align_img_to_seq = getattr(configs, 'align_img_to_seq', True)
        self.visual_encoder = SpatioTemporalEncoder(configs.d_model, in_ch=img_ch)
        self.cross_attn = nn.MultiheadAttention(configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True)
       
        self.cross_gate = nn.Parameter(torch.tensor(1.5, dtype=torch.float))
       
        self.mod_drop = getattr(configs, 'mod_drop', 0.0)

    def _interp_img_time(self, x_img, target_len):
        
        b, t, c, h, w = x_img.shape
        if t == target_len:
            return x_img
        x_flat = x_img.reshape(b, t, -1).permute(0, 2, 1)  
        x_interp = F.interpolate(x_flat, size=target_len, mode='linear', align_corners=False)
        x_interp = x_interp.permute(0, 2, 1).reshape(b, target_len, c, h, w)
        return x_interp

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_img=None):
        if self.use_norm:
            
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        
        enc_time = self.enc_embedding(x_enc, x_mark_enc) 
        enc_out, attns = self.encoder(enc_time, attn_mask=None)

       
        if x_img is not None and self.use_visual and self.training and self.mod_drop > 0:
            b_drop = x_img.shape[0]
            drop_mask = (torch.rand(b_drop, device=x_img.device) < self.mod_drop)
            if drop_mask.any():
                if drop_mask.all():
                    x_img = None
                else:
                    x_img = x_img.clone()
                    x_img[drop_mask] = 0.0

       
        if x_img is not None and self.use_visual:
            
            if x_img.dim() == 4: 
                x_img = x_img.unsqueeze(2)
            elif x_img.dim() == 3:  
                x_img = x_img.unsqueeze(0).unsqueeze(0)
            elif x_img.dim() != 5:
                raise ValueError(f"Unexpected x_img dim {x_img.shape}")

            
            if self.align_img_to_seq and x_img.shape[1] != self.seq_len and x_img.shape[1] % self.seq_len == 0:
                factor = x_img.shape[1] // self.seq_len
                b, l, c, h, w = x_img.shape
                x_img = x_img.view(b, self.seq_len, factor, c, h, w).mean(dim=2)

            
            img_feat = self.visual_encoder(x_img)
            cross_out, _ = self.cross_attn(query=enc_out, key=img_feat, value=img_feat)
            gate = torch.sigmoid(self.cross_gate)
            enc_out = enc_out + gate * cross_out
        
        
        dec_out = self.projector(enc_out).permute(0, 2, 1)  

        if self.use_norm:
            
            feat_dim = dec_out.shape[-1]
            std_src = stdev[:, 0, :]
            mean_src = means[:, 0, :]
            if std_src.shape[-1] < feat_dim:
               
                pad = feat_dim - std_src.shape[-1]
                std_src = F.pad(std_src, (0, pad))
                mean_src = F.pad(mean_src, (0, pad))
            std_slice = std_src[:, :feat_dim].unsqueeze(1)
            mean_slice = mean_src[:, :feat_dim].unsqueeze(1)
            dec_out = dec_out * std_slice
            dec_out = dec_out + mean_slice

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_img=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_img=x_img)
        
        
        if getattr(self, 'output_split', False):
            return dec_out

        final_pred = dec_out[:, -self.pred_len:, :]
        final_pred = torch.nn.functional.relu(final_pred)
        if self.output_attention:
            return final_pred, attns
        else:
            return final_pred


class PIModel(nn.Module):
    def __init__(self, base, configs):
        super(PIModel, self).__init__()
        self.base = base
        self.args = configs
        
        
        if not getattr(self.base, 'output_split', False):
            self.base.projector = nn.Linear(configs.d_model, configs.pred_len * 2, bias=True)
            self.base.output_split = True
            
        self.scaling_factor = float(getattr(configs, 'k_scale', 1.2)) 
        self.latitude = float(getattr(configs, 'latitude', 38.0))
        self.eta = float(getattr(configs, 'eta_sys', 1.0))
        self.capacity = float(getattr(configs, 'capacity', 1.0))
        self.op_start = float(getattr(configs, 'op_start', 6.75))
        self.op_end = float(getattr(configs, 'op_end', 18.75))
        self.temp_coef = float(getattr(configs, 'temp_coef', -0.004))
        self.std_temp = float(getattr(configs, 'std_temp', 25.0))
        self.tz_offset = float(getattr(configs, 'tz_offset', 8.0))
        self.day_mask_th = float(getattr(configs, 'day_mask_th', 0.05))
        self.resid_gamma = float(getattr(configs, 'resid_gamma', 0.1))
        self.pmax_floor = float(getattr(configs, 'pmax_floor', 0.2))
        self.p_norm_mode = str(getattr(configs, 'p_norm_mode', 'win'))
        self.res_scale = float(getattr(configs, 'res_scale', 0.15))
        self.cap_boost = float(getattr(configs, 'cap_boost', 1.0))
        self.smooth_k = int(getattr(configs, 'smooth_k', 3))
        self.mask_tau = float(getattr(configs, 'mask_tau', 0.02))
        self.taper_len = int(getattr(configs, 'taper_len', 12))
        self.k_head = nn.Linear(1, 1)
        self.res_head = nn.Linear(1, 1)
        self.target_mean = None
        self.target_std = None
        self.capacity = float(getattr(configs, 'capacity', 1.0))
        self.alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float))

    def _smooth1d(self, x, k):
       
        if k <= 1:
            return x
        pad = k // 2
        x_pad = F.pad(x.permute(0, 2, 1), (pad, pad), mode='replicate')
        x_s = F.avg_pool1d(x_pad, kernel_size=k, stride=1, padding=0)
        return x_s.permute(0, 2, 1)

    def _taper_edges(self, x):
    
        n = min(self.taper_len, x.shape[1] // 2)
        if n <= 0:
            return x
        left_ref = x[:, n:n+1, :]
        right_ref = x[:, -n-1:-n, :]
        alpha = torch.linspace(0, 1, steps=n, device=x.device, dtype=x.dtype).view(1, n, 1)
        left = x[:, :n, :] * (1 - alpha) + left_ref * alpha
        right = x[:, -n:, :] * alpha + right_ref * (1 - alpha)
        mid = x[:, n:-n, :] if x.shape[1] > 2 * n else x[:, 0:0, :]
        return torch.cat([left, mid, right], dim=1)

    def _p_theory(self, x_mark_dec):
        B, L, D = x_mark_dec.shape
        x_mark_dec = x_mark_dec[:, -self.args.pred_len:, :]
        if getattr(self.args, 'embed', 'timeF') == 'timeF':
           
            hour = x_mark_dec[:, :, -3].float()
            time_hour_idx = x_mark_dec[:, :, -1].float()
            doy = time_hour_idx / 24.0
        else:
            D = x_mark_dec.shape[-1]
            hour = x_mark_dec[:, :, 3].float()
            if D >= 5:
                time_hour_idx = x_mark_dec[:, :, 4].float()
                doy = time_hour_idx / 24.0
            else:
                month = x_mark_dec[:, :, 0].float()
                day = x_mark_dec[:, :, 1].float()
                month_days = torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], device=x_mark_dec.device, dtype=torch.float)
                mi = torch.clamp(month - 1, 0, 11).long()
                doy_day = month_days[mi] + day
                doy = doy_day - 1.0 + hour / 24.0
        lat_rad = torch.tensor(math.radians(self.latitude), device=hour.device)
        delta = 23.45 * torch.sin(torch.deg2rad(360.0 * (284.0 + doy) / 365.25))
        delta_rad = torch.deg2rad(delta)
        B = torch.deg2rad(360.0 * (doy - 81.0) / 365.0)
        eot = 9.87 * torch.sin(2.0 * B) - 7.53 * torch.cos(B) - 1.5 * torch.sin(B)
        lstm = 15.0 * self.tz_offset
        lst = hour + (eot + 4.0 * (self.args.longitude - lstm)) / 60.0
        omega = 15.0 * (lst - 12.0)
        omega_rad = torch.deg2rad(omega)
        cos_z = (torch.sin(lat_rad) * torch.sin(delta_rad) +
                 torch.cos(lat_rad) * torch.cos(delta_rad) * torch.cos(omega_rad))
        cos_z = torch.clamp(cos_z, min=0.0)
        t_cell = self.std_temp + 15.0 * cos_z
        eff = 1.0 + self.temp_coef * (t_cell - self.std_temp)
        eff = torch.clamp(eff, 0.5, 1.2)
        op_mask = ((hour >= self.op_start) & (hour <= self.op_end)).float()
        p_theory = self.capacity * self.eta * (cos_z ** 1.15) * eff * op_mask
        return p_theory.unsqueeze(-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_img=None):
        out = self.base(x_enc, x_mark_enc, x_dec, x_mark_dec, x_img=x_img)
        if isinstance(out, tuple):
            out = out[0]
            
        
        split_len = self.args.pred_len
        kc_logit = out[:, :split_len, :]
        res_val = out[:, split_len:, :]
        
       
        p_phy = self._p_theory(x_mark_dec) 
        
        p_phy_scaled = (p_phy / self.capacity) * self.cap_boost
        p_phy_scaled = self._smooth1d(p_phy_scaled, self.smooth_k)
        p_phy_scaled = self._taper_edges(p_phy_scaled)
            
        p_saturated = p_phy_scaled * self.scaling_factor
        
       
        kc_logit_target = kc_logit[:, :, -1:] 
        res_target = res_val[:, :, -1:]       
        
        res_others = res_val[:, :, :-1]       
        
       
        k_c_target = torch.sigmoid(kc_logit_target)
        
        k_c_target = self._smooth1d(k_c_target, self.smooth_k)
        res_target = self._smooth1d(res_target, self.smooth_k)
        
        k_c_target = self._taper_edges(k_c_target)
        res_target = self._taper_edges(res_target)

    
        day_th = float(getattr(self.args, 'day_mask_th', 0.05))
        tau = self.mask_tau if self.mask_tau > 0 else 0.02
        mask_day = torch.sigmoid((p_phy_scaled - day_th) / tau)
        k_c_target = k_c_target * mask_day
        res_target = res_target * mask_day

        pred_target = p_saturated * k_c_target + res_target
        pred_target = self._taper_edges(pred_target)
        
       
        pred = torch.cat([res_others, pred_target], dim=-1)

       
        self.last_phy = p_phy_scaled

        return torch.nn.functional.relu(pred)
    def set_target_norm(self, mean, std):
        self.target_mean = float(mean)
        self.target_std = float(std) if std != 0 else 1.0
    def set_capacity(self, cap):
        self.capacity = float(cap)
