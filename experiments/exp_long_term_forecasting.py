from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import math

warnings.filterwarnings('ignore')


class DaytimeMaskedHuber(nn.Module):
    def __init__(self, delta, th):
        super(DaytimeMaskedHuber, self).__init__()
        self.delta = float(delta)
        self.th = float(th)
    def forward(self, pred, target):
        mask = (target > self.th).float()
        diff = pred - target
        absd = torch.abs(diff)
        delta = self.delta
        loss = torch.where(absd <= delta, 0.5 * diff * diff, delta * (absd - 0.5 * delta))
        loss = loss * mask
        denom = torch.sum(mask)
        if denom.item() == 0:
            return torch.mean(0.5 * diff * diff)
        return torch.sum(loss) / denom

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        base = self.model_dict[self.args.model].Model(self.args).float()
        if bool(getattr(self.args, 'use_physics', False)) or bool(getattr(self.args, 'phys_hard', False)):
            from model.iTransformer import PIModel
            model = PIModel(base, self.args).float()
        else:
            model = base

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=getattr(self.args, 'weight_decay', 0.0))
        return model_optim

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'MSE')).upper()
        if loss_name == 'HUBER':
            return DaytimeMaskedHuber(delta=float(getattr(self.args, 'huber_delta', 0.1)), th=float(getattr(self.args, 'day_mask_th', 0.0)))
        return nn.MSELoss()

    def _phase_corr_loss(self, pred, y_mark):
        if y_mark is None:
            return torch.zeros(1, device=self.device)
        if getattr(self.args, 'embed', 'timeF') == 'timeF':
            return torch.zeros(1, device=self.device)
        B = pred.shape[0]
        L = pred.shape[1]
        hour = y_mark[:, -L:, 3]
        phase = 2.0 * math.pi * (hour % 24.0) / 24.0
        cos_b = torch.cos(phase).unsqueeze(-1)
        sin_b = torch.sin(phase).unsqueeze(-1)
        cos_p = torch.relu(cos_b)
        sin_p = torch.relu(sin_b)
        X = torch.cat([cos_p, sin_p], dim=-1)
        M = (cos_p > 0.01).float()
        eps = 1e-6
        Xt = X.transpose(1, 2)
        W = M
        XtWX = torch.bmm(Xt, X * W)
        I = torch.eye(2, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        XtWX = XtWX + eps * I
        XtWy = torch.bmm(Xt, pred * W)
        coef = torch.linalg.solve(XtWX, XtWy)
        y_hat = torch.bmm(X, coef)
        resid = (pred - y_hat) * W
        var = torch.mean((pred - torch.mean(pred, dim=1, keepdim=True)) ** 2, dim=1, keepdim=True) + eps
        fit_loss = torch.mean((resid ** 2) / var)
        nonneg = torch.mean(torch.relu(-pred))
        return fit_loss + 0.1 * nonneg

    def _clear_sky_power(self, y_mark, out_len=None):
        if y_mark is None:
            return None
        B, T, D = y_mark.shape
        if out_len is not None and T >= out_len:
            y_mark = y_mark[:, -out_len:, :]
        if getattr(self.args, 'embed', 'timeF') == 'timeF':
            hour = y_mark[:, :, -3].float()
            time_hour_idx = y_mark[:, :, -1].float()
            doy = time_hour_idx / 24.0
        else:
            D = y_mark.shape[-1]
            hour = y_mark[:, :, 3].float()
            if D >= 5:
                time_hour_idx = y_mark[:, :, 4].float()
                doy = time_hour_idx / 24.0
            else:
                month = y_mark[:, :, 0].float()
                day = y_mark[:, :, 1].float()
                month_days = torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], device=self.device, dtype=torch.float)
                mi = torch.clamp(month - 1, 0, 11).long()
                doy_day = month_days[mi] + day
                doy = doy_day - 1.0 + hour / 24.0
        lat = float(getattr(self.args, 'latitude', 38.0))
        eta = float(getattr(self.args, 'eta_sys', 1.0))
        cap = float(getattr(self.args, 'capacity', 1.0))
        op_start = float(getattr(self.args, 'op_start', 6.75))
        op_end = float(getattr(self.args, 'op_end', 18.75))
        temp_coef = float(getattr(self.args, 'temp_coef', -0.004))
        std_temp = float(getattr(self.args, 'std_temp', 25.0))
        lat_rad = torch.tensor(math.radians(lat), device=self.device)
        delta = 23.45 * torch.sin(torch.deg2rad(360.0 * (284.0 + doy) / 365.25))
        delta_rad = torch.deg2rad(delta)
        B = torch.deg2rad(360.0 * (doy - 81.0) / 365.0)
        eot = 9.87 * torch.sin(2.0 * B) - 7.53 * torch.cos(B) - 1.5 * torch.sin(B)
        lstm = 15.0 * float(getattr(self.args, 'tz_offset', 8.0))
        lst = hour + (eot + 4.0 * (float(getattr(self.args, 'longitude', 114.0)) - lstm)) / 60.0
        omega = 15.0 * (lst - 12.0)
        omega_rad = torch.deg2rad(omega)
        cos_z = (torch.sin(lat_rad) * torch.sin(delta_rad) +
                 torch.cos(lat_rad) * torch.cos(delta_rad) * torch.cos(omega_rad))
        cos_z = torch.clamp(cos_z, min=0.0)
        t_cell = std_temp + 15.0 * cos_z
        eff = 1.0 + temp_coef * (t_cell - std_temp)
        eff = torch.clamp(eff, 0.5, 1.2)
        op_mask = ((hour >= op_start) & (hour <= op_end)).float()
        p_theory = cap * eta * (cos_z ** 1.15) * eff * op_mask
        return p_theory.unsqueeze(-1)

    def _physics_loss(self, pred, y_mark, label=None):
        L = pred.shape[1]
        p_theory = self._clear_sky_power(y_mark, out_len=L)
        if p_theory is None:
            return torch.zeros(1, device=self.device)
            
     
        cap = float(getattr(self.args, 'capacity', 1.0))
        p_theory_norm = p_theory / cap
            
        eps = 1e-6
        cap = float(getattr(self.args, 'capacity', 1.0))
        eta = float(getattr(self.args, 'eta_sys', 1.0))
        p_max = cap * eta + eps
        th = float(getattr(self.args, 'day_mask_th', 0.05))
        mask_day = (p_theory.squeeze(-1) > th).float()
        if label is not None:
            lbl_mask = (label > 0).float()
            if lbl_mask.shape[-1] == 1:
                mask_day = mask_day * lbl_mask
        scale = float(getattr(self.args, 'k_scale', 1.2))
        
        # Only apply physics loss to the target channel (last channel)
        pred_target = pred[:, :, -1:]
        
        # Use normalized p_theory for comparison with normalized pred
        upper_violation = torch.relu(pred_target - scale * p_theory_norm) * mask_day.unsqueeze(-1)
        lower_violation = torch.relu(-pred_target)
        
        w = (p_theory / p_max).detach()
        denom = torch.sum(mask_day) + eps
        
        # Loss is already in normalized scale, so no need to divide by p_mean (absolute)
        loss_upper = torch.sum(w * upper_violation) / denom
        loss_lower = 0.1 * torch.mean(lower_violation)
        return loss_upper + loss_lower

    def _k_smooth_loss(self, pred, x_mark_dec):
        try:
            eps = 1e-6
            p = self.model._p_theory(x_mark_dec)
            k_est = pred / (p + eps)
            diff = k_est[:, 1:, :] - k_est[:, :-1, :]
            return torch.mean(diff ** 2)
        except Exception:
            return torch.zeros(1, device=self.device)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_img = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_img = None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_img is not None:
                    batch_img = batch_img.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
              
                if outputs.shape[-1] > batch_y.shape[-1]:
                    outputs = outputs[..., :batch_y.shape[-1]]

               
                out_use = outputs[:, :batch_y.shape[1], :batch_y.shape[-1]]
                tgt = batch_y[:, -out_use.shape[1]:, :out_use.shape[-1]]
                loss = criterion(out_use, tgt)
                lambda_phase = float(getattr(self.args, 'lambda_phase', 0.0))
                if lambda_phase > 0.0 and batch_y_mark is not None:
                    phase_loss = self._phase_corr_loss(outputs, batch_y_mark)
                    loss = loss + lambda_phase * phase_loss
                lambda_phys = float(getattr(self.args, 'lambda_phys', 0.0))
                if lambda_phys > 0.0 and batch_y_mark is not None:
                    loss = loss + lambda_phys * self._physics_loss(outputs, batch_y_mark)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if setting is None:
            setting = self.args.model_id
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # inject dataset target normalization and capacity to model for physics normalization
        try:
            if hasattr(train_data, 'scaler'):
                if hasattr(train_data.scaler, 'mean_') and hasattr(train_data.scaler, 'scale_'):
                    mu = float(train_data.scaler.mean_[-1])
                    sigma = float(train_data.scaler.scale_[-1])
                    self.norm_min = mu
                    self.norm_range = sigma
                    if hasattr(self.model, 'set_target_norm'):
                        self.model.set_target_norm(mu, sigma)
                    print(f"Normalization injection (StandardScaler): mu={mu:.4f}, sigma={sigma:.4f}")
                elif hasattr(train_data.scaler, 'data_max_') and hasattr(train_data.scaler, 'data_min_'):
                    # MinMaxScaler support
                    min_val = float(train_data.scaler.data_min_[-1])
                    max_val = float(train_data.scaler.data_max_[-1])
                    self.norm_min = min_val
                    self.norm_range = max_val - min_val
                    if hasattr(self.model, 'set_target_norm'):
                        # Pass min as mean, and range (max-min) as std for compatibility with (x-mean)/std formula
                        # The previous code used p_phy / std. Here we want (p_phy - min) / (max - min).
                        # If min is 0, it becomes p_phy / max.
                        self.model.set_target_norm(min_val, max_val - min_val)
                    print(f"Normalization injection (MinMaxScaler): min={min_val:.4f}, max={max_val:.4f}")
                        
            if hasattr(train_data, 'capacity_max') and train_data.capacity_max is not None:
                if hasattr(self.model, 'set_capacity'):
                    self.model.set_capacity(train_data.capacity_max)
                print(f"Capacity injection: {train_data.capacity_max}")
        except Exception as e:
            print(f"Normalization injection failed: {e}")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_img = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_img = None
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_img is not None:
                    batch_img = batch_img.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if float(getattr(self.args, 'w_day', 0.0)) > 0.0:
                            loss = self._weighted_mse(outputs, batch_y, batch_y_mark)
                        else:
                            out_use = outputs[:, :batch_y.shape[1], :batch_y.shape[-1]]
                            tgt = batch_y[:, -out_use.shape[1]:, :out_use.shape[-1]]
                            loss = criterion(out_use, tgt)
                        lambda_phase = float(getattr(self.args, 'lambda_phase', 0.0))
                        if lambda_phase > 0.0:
                            loss = loss + lambda_phase * self._phase_corr_loss(outputs, batch_y_mark)
                        lambda_phys = float(getattr(self.args, 'lambda_phys', 0.0))
                        if lambda_phys > 0.0:
                             phys_l = self._physics_loss(outputs, batch_y_mark, batch_y)
                             print(f"\tDEBUG: phys_loss={phys_l.item():.4f}")
                             loss = loss + lambda_phys * phys_l
                        lambda_k = float(getattr(self.args, 'lambda_k_smooth', 0.0))
                        if lambda_k > 0.0:
                            loss = loss + lambda_k * self._k_smooth_loss(outputs, batch_y_mark)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if float(getattr(self.args, 'w_day', 0.0)) > 0.0:
                        loss = self._weighted_mse(outputs, batch_y, batch_y_mark)
                    else:
                        out_use = outputs[:, :batch_y.shape[1], :batch_y.shape[-1]]
                        tgt = batch_y[:, -out_use.shape[1]:, :out_use.shape[-1]]
                        loss = criterion(out_use, tgt)
                    lambda_phase = float(getattr(self.args, 'lambda_phase', 0.0))
                    if lambda_phase > 0.0:
                        loss = loss + lambda_phase * self._phase_corr_loss(outputs, batch_y_mark)
                    lambda_phys = float(getattr(self.args, 'lambda_phys', 0.0))
                    if lambda_phys > 0.0:
                        loss = loss + lambda_phys * self._physics_loss(outputs, batch_y_mark, batch_y)
                    lambda_k = float(getattr(self.args, 'lambda_k_smooth', 0.0))
                    if lambda_k > 0.0:
                        loss = loss + lambda_k * self._k_smooth_loss(outputs, batch_y_mark)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            if setting is None:
                setting = self.args.model_id
            ckpt_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                print('loading model')
                self.model.load_state_dict(torch.load(ckpt_path))
            else:
                print(f'checkpoint not found: {ckpt_path}, skip loading and use current weights')

        preds = []
        trues = []
        phys = []
        if setting is None:
            setting = self.args.model_id
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_img = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_img = None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_img is not None:
                    batch_img = batch_img.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # grab physical envelope if available
                phy = None
                m = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(m, 'last_phy'):
                    try:
                        phy = m.last_phy[:, -self.args.pred_len:, :]
                        phy = phy[:, :, f_dim:]
                    except Exception:
                        phy = None

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if phy is not None:
                    phy = phy.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape  # (B, L, C)
                    flat_out = outputs.reshape(-1, shape[-1])
                    flat_y = batch_y.reshape(-1, shape[-1])
                    outputs = test_data.inverse_transform(flat_out).reshape(shape)
                    batch_y = test_data.inverse_transform(flat_y).reshape(shape)
                    # Skip inverse_transform on phys to avoid shape/broadcast issues; keep normalized

                pred = outputs
                true = batch_y
                if phy is not None:
                    phys.append(phy)

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

       
        tgt_channels = min(trues.shape[-1], preds.shape[-1])
        preds_eval = preds[..., :tgt_channels]
        trues_eval = trues[..., :tgt_channels]

        mae, mse, rmse, mape, mspe = metric(preds_eval, trues_eval)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

       
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds_eval)
        np.save(folder_path + 'true.npy', trues_eval)
        if len(phys) > 0:
            phys = np.array(phys).reshape(-1, phys[0].shape[-2], phys[0].shape[-1])
            np.save(folder_path + 'pred_phy.npy', phys)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_img = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_img = None
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_img is not None:
                    batch_img = batch_img.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_img=batch_img)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    def _weighted_mse(self, outputs, batch_y, batch_y_mark):
        if batch_y_mark is None:
            return torch.mean((outputs - batch_y) ** 2)
        L = outputs.shape[1]
        p_theory = self._clear_sky_power(batch_y_mark, out_len=L)
        if p_theory is None:
            return torch.mean((outputs - batch_y) ** 2)
        th = float(getattr(self.args, 'day_mask_th', 0.05))
        w_day = float(getattr(self.args, 'w_day', 0.0))
        mask_day = (p_theory.squeeze(-1) > th).float()
        err = (outputs - batch_y) ** 2
        w = 1.0 + w_day * mask_day.unsqueeze(-1)
        return torch.mean(err * w)
