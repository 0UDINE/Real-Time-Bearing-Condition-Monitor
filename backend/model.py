"""
Model architecture for the CWRU bearing-fault classifier.

This file mirrors the EndToEndSTFTClassifier used by notebooks/cwru_5channel.ipynb.
The model expects raw, order-tracked vibration at 12 kHz and builds a
5-channel spectrogram internally:

    1. raw STFT
    2. full-band envelope STFT
    3. envelope STFT of 300-1500 Hz
    4. envelope STFT of 1500-3000 Hz
    5. envelope STFT of 3000-5500 Hz

The band-pass filters are part of the model state, so the checkpoint contains
both the learned weights and the multiband filter kernel.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import firwin


@dataclass
class ModelConfig:
    """Hyperparameters required to reconstruct the training architecture."""

    num_classes: int = 4
    class_names: tuple = ("normal", "outer race", "inner race", "ball")

    segment_length: int = 4096
    segment_hop: int = 2048
    n_fft: int = 256
    stft_hop: int = 64
    stft_win_length: int = 256
    window_chunk_size: int = 128

    fault_sample_rate_hz: int = 12000
    bandpass_taps: int = 129
    band_edges: tuple = (
        (300, 1500),
        (1500, 3000),
        (3000, 5500),
    )

    augment: bool = True
    noise_std: float = 0.02
    max_shift_fraction: float = 0.10
    freq_mask_fraction: float = 0.15
    time_mask_fraction: float = 0.15


class EndToEndSTFTClassifier(nn.Module):
    """Classify raw, variable-length DE recordings with internal STFT preprocessing.

    Input:
        Raw waveform already order-tracked to the reference RPM.

    Eval output:
        Log-probabilities over the four classes, using attention-weighted
        pooling over segment-level predictions.
    """

    def __init__(self, config: ModelConfig, band_edges=None):
        super().__init__()
        if band_edges is None:
            band_edges = config.band_edges
        band_edges = tuple((float(low), float(high)) for low, high in band_edges)

        segment_length = config.segment_length
        stft_win_length = config.stft_win_length
        if config.segment_hop <= 0 or segment_length < stft_win_length:
            raise ValueError("Check segment and STFT sizes.")

        self.config = config
        self.segment_length = segment_length
        self.segment_hop = config.segment_hop
        self.n_fft = config.n_fft
        self.stft_hop = config.stft_hop
        self.stft_win_length = stft_win_length
        self.window_chunk_size = config.window_chunk_size
        self.augment = config.augment
        self.noise_std = config.noise_std
        self.max_shift_fraction = config.max_shift_fraction
        self.freq_mask_fraction = config.freq_mask_fraction
        self.time_mask_fraction = config.time_mask_fraction
        self.band_edges = band_edges

        self.register_buffer("stft_window", torch.hann_window(stft_win_length))

        self.num_bands = len(band_edges)
        band_kernels = [
            firwin(
                config.bandpass_taps,
                [low_hz, high_hz],
                fs=config.fault_sample_rate_hz,
                pass_zero=False,
            )
            for low_hz, high_hz in band_edges
        ]

        # [num_bands, 1, taps]. A single conv1d call applies all band filters.
        self.register_buffer(
            "multiband_kernel",
            torch.tensor(
                np.stack(band_kernels), dtype=torch.float32
            ).unsqueeze(1),
        )

        # raw + full-band envelope + one envelope per band
        total_channels = 2 + self.num_bands
        self.spectrogram_cnn = nn.Sequential(
            nn.Conv2d(total_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(128, config.num_classes),
        )
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _segment_recordings(self, raw, lengths):
        if raw.size(1) < self.segment_length:
            raw = torch.nn.functional.pad(
                raw, (0, self.segment_length - raw.size(1))
            )
        segments = raw.unfold(-1, self.segment_length, self.segment_hop)
        valid_counts = (
            (lengths - self.segment_length) // self.segment_hop + 1
        ).clamp(min=0)
        valid_mask = (
            torch.arange(segments.size(1), device=raw.device)[None, :]
            < valid_counts[:, None]
        )
        return segments, valid_mask

    def _augment_raw(self, segments):
        if self.max_shift_fraction > 0:
            max_shift = max(
                1, int(self.segment_length * self.max_shift_fraction)
            )
            shifts = torch.randint(
                -max_shift,
                max_shift + 1,
                (segments.size(0),),
                device=segments.device,
            )
            segments = torch.stack(
                [torch.roll(s, int(sh.item())) for s, sh in zip(segments, shifts)]
            )
        return segments

    def _augment_spectrogram(self, spec):
        n_freq, n_time = spec.size(2), spec.size(3)
        if self.freq_mask_fraction > 0:
            f_width = max(1, int(n_freq * self.freq_mask_fraction))
            f_start = random.randint(0, max(0, n_freq - f_width))
            spec[:, :, f_start : f_start + f_width, :] = 0.0
        if self.time_mask_fraction > 0:
            t_width = max(1, int(n_time * self.time_mask_fraction))
            t_start = random.randint(0, max(0, n_time - t_width))
            spec[:, :, :, t_start : t_start + t_width] = 0.0
        return spec

    def _multiband_filter(self, segments):
        # [n, length] -> [n, num_bands, length]
        x = segments.unsqueeze(1)
        pad = self.multiband_kernel.size(-1) // 2
        filtered = torch.nn.functional.conv1d(
            x, self.multiband_kernel, padding=pad
        )
        return filtered[..., : segments.size(-1)]

    def _envelope_raw(self, filtered_or_raw):
        n = filtered_or_raw.size(-1)
        spectrum = torch.fft.fft(filtered_or_raw, dim=-1)
        h = torch.zeros(
            n,
            device=filtered_or_raw.device,
            dtype=filtered_or_raw.dtype,
        )
        if n % 2 == 0:
            h[0] = 1.0
            h[1 : n // 2] = 2.0
            h[n // 2] = 1.0
        else:
            h[0] = 1.0
            h[1 : (n + 1) // 2] = 2.0
        analytic = torch.fft.ifft(spectrum * h, dim=-1)
        envelope = analytic.abs()
        return envelope - envelope.mean(dim=-1, keepdim=True)

    def _envelope_fullband(self, segments):
        return self._envelope_raw(segments)

    def _stft_channel(self, segments):
        stft = torch.stft(
            segments,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_win_length,
            window=self.stft_window,
            center=False,
            return_complex=True,
        )
        return torch.log1p(stft.abs()).unsqueeze(1)

    def _stft_multi(self, x):
        # [n, num_bands, length] -> [n, num_bands, freq, time]
        n, b, length = x.shape
        flat = x.reshape(n * b, length)
        spec = self._stft_channel(flat).squeeze(1)
        return spec.view(n, b, spec.size(-2), spec.size(-1))

    def segment_to_log_stft(self, segments):
        # Exact normalization used by the training notebook.
        segments = (
            segments - segments.mean(dim=-1, keepdim=True)
        ) / segments.std(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.training and self.noise_std > 0:
            segments = segments + torch.randn_like(segments) * self.noise_std

        raw_spec = self._stft_channel(segments)

        env_full = self._envelope_fullband(segments)
        env_full = env_full / env_full.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        env_full_spec = self._stft_channel(env_full)

        band_filtered = self._multiband_filter(segments)
        band_envelopes = self._envelope_raw(band_filtered)
        band_envelopes = band_envelopes / band_envelopes.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        band_spec = self._stft_multi(band_envelopes)

        # [raw, full-band envelope, 3 band envelopes] => 5 channels
        spec = torch.cat([raw_spec, env_full_spec, band_spec], dim=1)

        if self.training:
            spec = self._augment_spectrogram(spec)
        return spec

    def forward(self, raw, lengths, return_segment_logits=False):
        segments, valid_mask = self._segment_recordings(raw, lengths)
        batch_size, n_segments, segment_length = segments.shape

        flat_segments = segments.reshape(-1, segment_length)
        flat_valid = valid_mask.reshape(-1)
        valid_segments = flat_segments[flat_valid]

        if self.training and self.max_shift_fraction > 0:
            valid_segments = self._augment_raw(valid_segments)

        spectrograms = self.segment_to_log_stft(valid_segments)

        features = []
        for spectrogram_chunk in spectrograms.split(self.window_chunk_size):
            chunk_features = self.spectrogram_cnn(spectrogram_chunk).flatten(1)
            features.append(chunk_features)
        valid_features = torch.cat(features)
        valid_logits = self.classifier(valid_features)

        if return_segment_logits:
            batch_index = torch.arange(batch_size, device=raw.device)
            batch_index = (
                batch_index.unsqueeze(1)
                .expand(batch_size, n_segments)
                .reshape(-1)[flat_valid]
            )
            return valid_logits, batch_index

        feat_dim = valid_features.size(-1)
        segment_features = torch.zeros(
            batch_size * n_segments,
            feat_dim,
            device=raw.device,
            dtype=valid_features.dtype,
        )
        segment_features[flat_valid] = valid_features
        segment_features = segment_features.view(
            batch_size, n_segments, feat_dim
        )

        segment_logits_full = torch.zeros(
            batch_size * n_segments,
            valid_logits.size(-1),
            device=raw.device,
            dtype=valid_logits.dtype,
        )
        segment_logits_full[flat_valid] = valid_logits
        segment_logits_full = segment_logits_full.view(
            batch_size, n_segments, -1
        )

        attn_scores = self.attention(segment_features).squeeze(-1)
        attn_scores = attn_scores.masked_fill(
            ~valid_mask, float("-inf")
        )
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)

        recording_logits = (
            segment_logits_full * attn_weights
        ).sum(dim=1)
        return torch.log_softmax(recording_logits, dim=-1)
