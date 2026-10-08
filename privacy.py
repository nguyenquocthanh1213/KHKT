"""
privacy.py - Ke toan quyen rieng tu (RDP accountant) cho DP-SGD voi Poisson sampling.

Tu chua (khong can Opacus). Cong thuc Sampled Gaussian Mechanism voi bac nguyen alpha
(Mironov, Talwar, Zhang 2019):
    A_a  = sum_{k=0..a} C(a,k) (1-q)^(a-k) q^k exp((k^2-k) / (2 sigma^2))
    RDP(a) = log(A_a) / (a-1)
RDP cong tuyen tinh qua cac buoc / cac round. Doi sang (epsilon, delta) bang cong thuc
chat hon cua Balle et al. 2020 (giong Opacus).
"""
import math
from pathlib import Path

import numpy as np

from common import load_json, save_json

ORDERS = list(range(2, 257))


def _log_a(q: float, sigma: float, alpha: int) -> float:
    if q >= 1.0:
        return alpha * (alpha - 1) / (2 * sigma ** 2)
    terms = []
    for k in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        t = log_binom + k * math.log(q) + (alpha - k) * math.log1p(-q) + (k * k - k) / (2 * sigma ** 2)
        terms.append(t)
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def compute_rdp(q: float, sigma: float, steps: int, orders=ORDERS):
    if q <= 0 or steps == 0:
        return [0.0] * len(orders)
    return [steps * _log_a(q, sigma, a) / (a - 1) for a in orders]


def rdp_to_eps(rdp, delta: float, orders=ORDERS) -> float:
    o = np.asarray(orders, dtype=float)
    r = np.asarray(rdp, dtype=float)
    eps = r - (math.log(delta) + np.log(o)) / (o - 1) + np.log((o - 1) / o)
    return float(np.nanmin(np.maximum(eps, 0.0)))


class ClientPrivacyLedger:
    """So cai rieng cua MOT client: luu RDP theo tung round -> epsilon tich luy qua cac round.
    (Chay lai cung mot round se ghi de, khong tinh trung.)"""

    def __init__(self, path):
        self.path = Path(path)
        self.rounds = load_json(self.path)["rounds"] if self.path.exists() else {}

    def record(self, round_idx: int, q: float, sigma: float, steps: int):
        self.rounds[str(round_idx)] = compute_rdp(q, sigma, steps)
        save_json({"orders": ORDERS, "rounds": self.rounds}, self.path)

    def epsilon(self, delta: float, extra=None) -> float:
        """epsilon tich luy. `extra` = (round, q, sigma, steps) de du bao truoc khi huan luyen."""
        rounds = dict(self.rounds)
        if extra is not None:
            r, q, sigma, steps = extra
            rounds[str(r)] = compute_rdp(q, sigma, steps)
        total = np.sum([np.asarray(v) for v in rounds.values()], axis=0)
        return rdp_to_eps(total, delta)
