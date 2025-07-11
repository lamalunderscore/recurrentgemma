"""Utility functions for dejavu implementation."""

from typing import Literal

import torch


def get_topk(
    attn_weights,
    k: int,
    metric="l2",
    do_prefill=False,
) -> torch.Tensor | None:
    """Return topk head indices for each token.

    Uses different metrics as a measure of uniformity. Return topk least-
    uniform heads for each token.

    Args:
        attn_weights (_type_): Weight matrix to calculate topk on.
        k (int): Number of indices to be returned.
        metric (str, optional): Metric to measure uniformity. Defaults to "l2".
        do_prefill (bool, optional): If the sparsification should be calculated during the
            prefill stage. Defaults to False.

    Raises:
        ValueError: If k is bigger than the number of heads, or k < 0.

    Returns:
        torch.Tensor | None: Tensor with topk indices for each token, if k != 0 else None

    """
    num_heads = attn_weights.shape[1]
    metric_map = {
        "l2": lambda x: torch.norm(x, p=2, dim=-1),
        "entropy": lambda x: torch.sum((x + 1e-12) * torch.log2(x + 1e-12), dim=-1),
    }

    out_len = attn_weights.shape[-2]
    if not do_prefill and not out_len == 1:
        k = num_heads  # deactivate sparsification

    if k == 0:
        return None

    if k > num_heads or k < 0:
        raise ValueError(f"k ({k}) cannot exceed number of attention heads ({num_heads})")
    metric_scores = metric_map[metric](attn_weights)

    _, topk_ind = metric_scores.topk(k, dim=1)

    return topk_ind


def keep_topk(attn_output, topk: torch.Tensor | None) -> torch.Tensor:
    """Mask out indices not mentioned in the topk tensor.

    Args:
        attn_output (_type_): Matrix to be masked.
        topk (torch.Tensor): Tensor including to-be-kept head indices for every token.

    Returns:
        torch.Tensor: Masked version of attn_output.

    """
    if topk is None:
        return attn_output * 0.0

    batch_size, target_length, num_heads, h = attn_output.shape
    batch_size, k, target_length = topk.shape
    if k == num_heads:
        return attn_output

    mask = torch.zeros(
        batch_size, target_length, num_heads, dtype=torch.bool, device=attn_output.device
    )

    # make dimensions fit mask tensor
    topk = torch.einsum("bks -> bsk", topk).to(device=attn_output.device)

    # unmask head indices in topk
    mask.scatter_(
        dim=2,  # k dimension
        index=topk,
        src=torch.ones_like(topk, dtype=torch.bool, device=attn_output.device),
    )
    attn_output = attn_output * mask.unsqueeze(dim=-1)
    return attn_output


def _init_value_params(
    token_indices: list[int], mode: Literal["ommit", "only", "balanced", "null"], probs
):
    if mode == "ommit":  # null needle
        return (0, 1)
    if mode == "null":  # null everything
        return (0, 0)
    if mode == "only":  # keep needle, null rest
        return (42, 0)  # first value is unused in this case, so I can show that I am a nerd.
    if mode == "balanced":  # null everything and then set needle to balanced softmax
        batch_size, num_heads, target_length, num_weights = probs.shape
        balanced_value = probs[..., token_indices].sum() / num_heads / len(token_indices)
        return (balanced_value, 0)


def manipulate_weights(
    probs: torch.Tensor,
    manipulation: tuple[list[int], Literal["ommit", "only", "balanced", "null"]],
    sliding_window_size: int,
    sequence_length: int,
):
    """Manipulate attention weights in different modes.

    Args:
        probs (torch.Tensor): The original softmax tensor.
        manipulation (tuple[list[int], Literal["ommit", "only", "balanced"]]):
            Tokens and modes to manipulate by.
            Gen modes:
            "Ommit": set specified tokens to 0.
            "Only": set every token but the specified tokens to 0.
            "balanced": set every token to 0 and calculate balanced weights on the specified tokens.
            Prefill modes:
            "Ommit": set all values to 0.
            "manipulate": follow gen mode.
            "keep": do not manipulate.
        sliding_window_size (int): Sliding window size of the model.
        sequence_length (int): Length of the input sequence. Necessary to compute position in circular buffer.
        prefill (bool, optional): Mode that prefill should be handled with.

    Returns:
        torch.Tensor: The manipulated sotmax tensor.

    """
    batch_size, num_heads, target_length, num_weights = probs.shape
    assert batch_size == 1, "Attention weight manipulation only works with batch_size = 1"
    assert isinstance(sequence_length, int)

    probs_manipulated = probs.clone()

    token_indices, mode = manipulation
    if target_length == 1:  # there is no sliding window during prefill
        token_indices = [
            token_index % sliding_window_size  # RG uses a circular cache buffer
            for token_index in token_indices
            if token_index >= sequence_length - sliding_window_size
        ]
        if not token_indices:
            return probs_manipulated

    token_value, non_token_weight = _init_value_params(token_indices, mode, probs)

    for weight_index in range(num_weights):
        if weight_index in token_indices and mode in [  # only change token weights in these modes
            "balanced",
            "ommit",
            "null",
        ]:
            probs_manipulated[..., weight_index] = token_value
        if weight_index not in token_indices:
            probs_manipulated[..., weight_index] *= non_token_weight

    return probs_manipulated


__all__ = ("get_topk", "keep_topk", "manipulate_weights")
