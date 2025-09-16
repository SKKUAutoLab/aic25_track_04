#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the symmetric power transformation in the paper: "Enhancing Implicit
Neural Representations via Symmetric Power Transformation," AAAI 2025.

References:
    - https://github.com/zwx-open/Symmetric-Power-Transformation-INR
"""

__all__ = [
    "SymPowerTransform",
]

from typing import Literal

import numpy as np
import torch
from scipy import stats


class SymPowerTransform:

    def __init__(
        self,
        method        : Literal["min_max", "z_score", "sym_power", "box_cox", "gamma"] = "sym_power",
        trans_shift   : float = -0.5,
        trans_scale   : float =  2.0,
        pn_cum        : float =  0.5,
        pn_buffer     : float = -1.0,
        box_shift     : float =  0.1,
        pn_beta       : float =  0.01,
        pn_k          : float =  256.0,
        pn_alpha      : float =  0.05,
        gamma_boundary: float =  5.0,
        gamma_trans   : float = -1.0,
    ):
        self.inverse_map    = {}
        self.method         = method
        self.trans_shift    = trans_shift
        self.trans_scale    = trans_scale
        self.trans_scale    = trans_scale
        self.pn_cum         = pn_cum
        self.pn_buffer      = pn_buffer
        self.box_shift      = box_shift
        self.pn_beta        = pn_beta
        self.pn_k           = pn_k
        self.pn_alpha       = pn_alpha
        self.gamma_boundary = gamma_boundary
        self.gamma_trans    = gamma_trans

    def transform(self, x: torch.tensor) -> torch.tensor:
        device = x.device
        x      = x.cpu()
        if self.method == "min_max":
            _min  = torch.min(x)
            _max  = torch.max(x)
            self.inverse_map[self.method] = (_min, _max)
            y     = (x - _min) / (_max - _min)
            y     = self._encode_shift_scale(y)
        elif self.method == "z_score":
            _mean = x.mean()
            _std  = x.std()
            y     = (x - _mean) / (_std + 1e-5)
            self.inverse_map[self.method] = (_mean, _std)
        elif self.method == "sym_power":
            _meta, y = self._sym_power_trans(x)
            self.inverse_map[self.method] = _meta
        elif self.method == "box_cox":
            _min, _max, _lambda, y = self._box_cox(x)
            self.inverse_map[self.method] = (_min, _max, _lambda)
        else:
            raise NotImplementedError
        return y.to(device)

    def inverse(self, y: torch.tensor) -> torch.tensor:
        device = y.device
        y      = y.cpu()
        if self.method == "min_max":
            # scale → shift
            y = self._decode_shift_scale(y)
            _min, _max = self.inverse_map[self.method]
            x = y * (_max - _min) + _min
        elif self.method == "z_score":
            _mean, _std = self.inverse_map[self.method]
            x = y * (_std + 1e-5) + _mean
        elif self.method == "sym_power":
            _meta = self.inverse_map[self.method]
            x = self._inverse_sym_power_trans(_meta, y)
        elif self.method == "box_cox":
            _min, _max, _lambda = self.inverse_map[self.method]
            x = self._inverse_box_cox(_min, _max, _lambda, y)
        else:
            raise NotImplementedError
        return x.to(device)

    def _encode_shift_scale(self, x: torch.tensor) -> torch.tensor:
        x = x + self.trans_shift
        x = x * self.trans_scale
        return x

    def _decode_shift_scale(self, x: torch.tensor) -> torch.tensor:
        x = x / self.trans_scale
        x = x - self.trans_shift
        return x

    def _sym_power_trans(self, x: torch.tensor) -> tuple:
        _min = torch.min(x)
        _max = torch.max(x)
        hist = torch.histc(x.flatten(), bins=256, min=x.min(), max=x.max())
        pdf  = hist / hist.sum()

        if self.pn_beta <= 0:
            gamma = self._get_gamma_by_half(x)
        else:
            gamma = self._get_gamma_by_edge_calibration(x)

        boundary = self.gamma_boundary
        if gamma > 1:
            gamma = min(boundary, gamma)
        else:
            gamma = max(1 / boundary, gamma)

        if self.pn_buffer < 0:  # adaptive
            _alpha_len       = int(self.pn_alpha * 256)
            left_alpha_sum   = pdf[:_alpha_len].sum()
            right_alpha_sum  = pdf[-_alpha_len:].sum()
            _left_shift_len  = self.pn_k * left_alpha_sum
            _right_shift_len = self.pn_k * right_alpha_sum
        else:
            _left_shift_len  = (_max - _min) * self.pn_buffer
            _right_shift_len = (_max - _min) * self.pn_buffer

        _shift_len = _left_shift_len + _right_shift_len

        # gamma transformation
        if self.gamma_trans >= 0:
            gamma = self.gamma_trans
            _left_shift_len = _right_shift_len = _shift_len = 0

        y = (x - (_min - _left_shift_len)) / (_max - _min + _shift_len)  # [0,1]
        y = torch.pow(y, gamma)
        y = self._encode_shift_scale(y)

        return [_min, _max, gamma, _left_shift_len, _shift_len], y

    def _inverse_sym_power_trans(self, _meta, y: torch.tensor) -> torch.tensor:
        # zero-mean → gamma → min-max
        y = self._decode_shift_scale(y)
        _min, _max, gamma, _left_shift_len, _shift_len = _meta
        # clip to (0,1)
        y = torch.clamp(y, min=0.0, max=1.0)
        y = torch.pow(y, 1.0 / gamma)
        x = y * (_max - _min + _shift_len) + (_min - _left_shift_len)
        return x

    def _get_gamma_by_half(self, x: torch.tensor) -> float:
        hist       = torch.histc(x.flatten(), bins=256, min=x.min(), max=x.max())
        pdf        = hist / hist.sum()
        cdf        = torch.cumsum(pdf, dim=0)
        half_index = torch.searchsorted(cdf, self.pn_cum)  # 0.5
        half_perc  = half_index / 256
        gamma      = np.log(0.5) / np.log(half_perc)
        return gamma

    def _get_gamma_by_edge_calibration(self, x: torch.tensor) -> float:
        """deviation-aware calibration"""
        half_gamma = self._get_gamma_by_half(x)
        hist = torch.histc(x.flatten(), bins=256, min=x.min(), max=x.max())
        pdf  = hist / hist.sum()

        min_max_normed    = (x - x.min()) / (x.max() - x.min())
        half_gamma_normed = torch.pow(min_max_normed, half_gamma)
        half_gamma_hist   = torch.histc(half_gamma_normed.flatten(), bins=256, min=half_gamma_normed.min(), max=half_gamma_normed.max())
        half_gamma_pdf    = half_gamma_hist / half_gamma_hist.sum()

        _beta_len         = int(self.pn_beta * 256)
        _minmax_bin       = pdf[:_beta_len] if half_gamma> 1 else pdf[-_beta_len:]
        _half_gamma_bin   = half_gamma_pdf[:_beta_len] if half_gamma > 1 else half_gamma_pdf[-_beta_len:]

        delta_sum  = _half_gamma_bin.sum() - _minmax_bin.sum()  # assert > 0
        delta_sum /= self.pn_beta
        if half_gamma < 1:
            delta_gamma = 1 / half_gamma - 1
        else:
            delta_gamma = half_gamma - 1

        new_delta_gamma = delta_gamma * min(delta_sum, 1)

        if half_gamma < 1:
            new_gamma = 1 / ( 1 / half_gamma - new_delta_gamma)
        else:
            new_gamma = half_gamma - new_delta_gamma

        return new_gamma

    def _box_cox(self, x: torch.tensor) -> tuple:
        # sshift → boxcox → min-max → zero_mean
        x          = x + self.box_shift * 256
        _shape     = x.shape
        x, _lambda = stats.boxcox(x.flatten())
        x          = torch.tensor(x)
        x          = x.reshape(_shape)
        _min       = torch.min(x)
        _max       = torch.max(x)
        x          = (x - _min) / (_max - _min)  # [0,1]
        y          = self._encode_shift_scale(x)
        return _min, _max, _lambda, y

    def _inverse_box_cox(self, _min, _max, _lambda, y: torch.tensor) -> torch.tensor:
        # zero_mean → min-max → boxcox → - shift
        y = self._decode_shift_scale(y)
        y = torch.clamp(y, min=0.0, max=1.0)
        y = y * (_max - _min) + _min
        if _lambda == 0:
            y = torch.exp(y)
        else:
            y = (_lambda * y + 1) ** (1 / _lambda)
        x = y - self.box_shift * 256
        return x
