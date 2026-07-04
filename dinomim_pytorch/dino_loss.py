"""
DINO self-distillation loss: teacher softmax (centered, temperature-scaled) vs student log-softmax.
Adapted from the local DINO_MIM reference (``utils.Loss``) with schedules and entropy-side metrics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(p * (p + eps).log()).sum(dim=-1).mean()


def _entropy_from_logits(logits: torch.Tensor, temp: float) -> torch.Tensor:
    p = F.softmax(logits / max(temp, 1e-8), dim=-1)
    return _entropy_from_probs(p)


class DINOLoss(nn.Module):
    """
    Cross-entropy between teacher probability and student log-probability, averaged over
    valid (teacher, student) pairs. Skips (t_ix, s_ix) when ``s_ix == t_ix`` and
    ``s_ix < n_teacher`` (aligned global crops; same as original DINO multi-crop).
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        teacher_temp_warmup: Optional[float] = None,
        warmup_teacher_temp_epochs: int = 0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.teacher_temp_warmup = teacher_temp if teacher_temp_warmup is None else float(teacher_temp_warmup)
        self.warmup_teacher_temp_epochs = int(warmup_teacher_temp_epochs)
        self.register_buffer("center", torch.zeros(1, out_dim))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def current_teacher_temp(self) -> float:
        if self.warmup_teacher_temp_epochs <= 0:
            return float(self.teacher_temp)
        w = self.teacher_temp_warmup
        t = self.teacher_temp
        e = min(self._epoch, self.warmup_teacher_temp_epochs)
        if self.warmup_teacher_temp_epochs == 0:
            return t
        alpha = e / max(self.warmup_teacher_temp_epochs, 1)
        return float(w + (t - w) * alpha)

    @torch.no_grad()
    def update_center(self, teacher_output: List[torch.Tensor]) -> None:
        if not teacher_output:
            return
        cat = torch.cat(teacher_output, dim=0)
        batch_center = cat.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        *,
        n_global_student: int,
        student_view_weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        ``n_global_student``: number of student *global* crops (first indices). Remaining are local.
        ``student_view_weights``: per-student-index weight (e.g. global 1.0, local 0.5). Defaults to 1.0.
        """
        n_t = len(teacher_logits)
        n_s = len(student_logits)
        if student_view_weights is None:
            student_view_weights = [1.0] * n_s
        t_temp = self.current_teacher_temp()
        s_temp = max(float(self.student_temp), 1e-8)

        center = self.center
        student_sm = [F.log_softmax(s / s_temp, dim=-1) for s in student_logits]
        teacher_raw = [(t - center) / t_temp for t in teacher_logits]
        teacher_sm = [F.softmax(tr, dim=-1).detach() for tr in teacher_raw]

        total_loss = student_logits[0].new_tensor(0.0)
        weighted_sum = 0.0
        global_loss = student_logits[0].new_tensor(0.0)
        local_loss = student_logits[0].new_tensor(0.0)
        n_glob_w = 0.0
        n_loc_w = 0.0

        n_sg = int(n_global_student)
        g_student = n_sg
        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                # Skip when teacher slot matches the paired student *global* slot (DINO same-view).
                if t_ix == s_ix and s_ix < n_sg:
                    continue
                w = float(student_view_weights[s_ix] if s_ix < len(student_view_weights) else 1.0)
                term = torch.sum(-t * s, dim=-1).mean() * w
                total_loss = total_loss + term
                weighted_sum += w
                if s_ix < g_student:
                    global_loss = global_loss + term
                    n_glob_w += w
                else:
                    local_loss = local_loss + term
                    n_loc_w += w

        if weighted_sum > 0:
            total_loss = total_loss / weighted_sum
        if n_glob_w > 0:
            global_loss = global_loss / n_glob_w
        else:
            global_loss = global_loss * 0.0
        if n_loc_w > 0:
            local_loss = local_loss / n_loc_w
        else:
            local_loss = local_loss * 0.0

        self.update_center(teacher_logits)

        t_ent = _entropy_from_probs(teacher_sm[0]) if teacher_sm else total_loss * 0.0
        s_ent = _entropy_from_logits(student_logits[0], s_temp) if student_logits else total_loss * 0.0
        center_norm = self.center.norm().item()

        meta = {
            "global_loss": float(global_loss.detach()),
            "local_loss": float(local_loss.detach()),
            "teacher_entropy": float(t_ent.detach()),
            "student_entropy": float(s_ent.detach()),
            "teacher_temp": t_temp,
            "student_temp": float(s_temp),
            "center_norm": float(center_norm),
        }
        return total_loss, meta
