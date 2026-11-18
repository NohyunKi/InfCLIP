"""InfCLIP score (Eq. (2) and Algorithm 1 of the paper).

Given L2-normalized train features (f_i, g_i) and target features (f_t, g_t), the score of a
training sample i with respect to a target set T is

    I(i, T) = max_t  p_ii p_tt (f_i.f_t)(gbar_i.gbar_t) + p_ii q_tt (f_i.fbar_t)(gbar_i.g_t)
                   + q_ii p_tt (fbar_i.f_t)(g_i.gbar_t) + q_ii q_tt (fbar_i.fbar_t)(g_i.g_t)

computed from the image-encoder part and the (identical) text-encoder part. The constant factor
2 / tau^2 of Eq. (2) is omitted because it does not change the ranking.
"""

import torch
from tqdm import tqdm


def normalizing_l2(x):
    return x / x.norm(dim=-1, keepdim=True)


def compute_softmax_stats(f, g, logit_scale):
    """
    f, g: (B, D) L2-normalized image / text features of one batch.

    Returns the diagonal softmax terms p_ii, q_ii and the centered features
    fbar_i = f_i - sum_k q_ki f_k  and  gbar_i = g_i - sum_k p_ik g_k.
    """
    logits = logit_scale * (f @ g.T)
    p = torch.softmax(logits, dim=1)   # p_{ij}: image-to-text
    q = torch.softmax(logits, dim=0)   # q_{ij}: text-to-image

    p_diag = torch.diagonal(p)
    q_diag = torch.diagonal(q)

    f_bar = f - q.T @ f
    g_bar = g - p @ g

    del p, q
    return p_diag, q_diag, f_bar, g_bar


def calculate_inf_softmax_ntk(
    train_features_outputs,
    target_features_outputs,
    batch_size,
    logit_scale,
    device,
):
    """
    Args:
        train_features_outputs: dict with 'image_outputs' (N, D) and 'text_outputs' (N, D).
        target_features_outputs: dict with 'image_outputs' (M, D) and 'text_outputs' (M, D),
            e.g. M = 1000 ImageNet class prototypes.
        batch_size: number of training samples per softmax normalization (paper: 1024).
        logit_scale: 1 / tau_score (paper: 10).

    Returns:
        numpy array of shape (N,) with InfCLIP scores (higher = more valuable).
    """
    with torch.no_grad():
        fi_total = normalizing_l2(train_features_outputs["image_outputs"].float())
        gi_total = normalizing_l2(train_features_outputs["text_outputs"].float())

        fv = normalizing_l2(target_features_outputs["image_outputs"].float()).to(device)
        gv = normalizing_l2(target_features_outputs["text_outputs"].float()).to(device)

        # Target-side statistics do not depend on the training batch: compute once.
        p_diag_v, q_diag_v, f_bar_v, g_bar_v = compute_softmax_stats(fv, gv, logit_scale)

        N = fi_total.shape[0]
        n_batches = (N + batch_size - 1) // batch_size
        influence_score = []

        # Training batches are formed in dataset order (no shuffling), as in the paper.
        for batch_num in tqdm(range(n_batches), desc="InfCLIP"):
            idx = slice(batch_num * batch_size, min(N, (batch_num + 1) * batch_size))

            fi = fi_total[idx].to(device)
            gi = gi_total[idx].to(device)

            p_diag_i, q_diag_i, f_bar_i, g_bar_i = compute_softmax_stats(fi, gi, logit_scale)

            w_pp = p_diag_i[:, None] * p_diag_v[None, :]
            w_pq = p_diag_i[:, None] * q_diag_v[None, :]
            w_qp = q_diag_i[:, None] * p_diag_v[None, :]
            w_qq = q_diag_i[:, None] * q_diag_v[None, :]

            # Image-encoder (F) part
            sim_F = (
                w_pp * ((g_bar_i @ g_bar_v.T) * (fi @ fv.T))
                + w_pq * ((g_bar_i @ gv.T) * (fi @ f_bar_v.T))
                + w_qp * ((gi @ g_bar_v.T) * (f_bar_i @ fv.T))
                + w_qq * ((gi @ gv.T) * (f_bar_i @ f_bar_v.T))
            )

            # Text-encoder (G) part
            sim_G = (
                w_pp * ((fi @ fv.T) * (g_bar_i @ g_bar_v.T))
                + w_pq * ((fi @ f_bar_v.T) * (g_bar_i @ gv.T))
                + w_qp * ((f_bar_i @ fv.T) * (gi @ g_bar_v.T))
                + w_qq * ((f_bar_i @ f_bar_v.T) * (gi @ gv.T))
            )

            sim = sim_F + sim_G

            # I(i, T) = max over target prototypes
            influence_score.append(torch.max(sim, dim=1).values.cpu())

        influence_score = torch.cat(influence_score, dim=0).numpy()

    return influence_score
