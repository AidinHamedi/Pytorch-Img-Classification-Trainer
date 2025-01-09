# Libs >>>
from pytorch_optimizer import agc

# Main >>>
def adaptive_gradient_clipping(model, agc_eps: float, agc_clip_val: float, eps: float = 1e-6, exclude_bias=True):
    r"""Apply adaptive gradient clipping to the model's parameters, excluding certain parameters if specified.

    :param model: torch.nn.Module. The model whose gradients are to be clipped.
    :param agc_eps: float. AGC epsilon to clip the norm of parameters.
    :param agc_clip_val: float. Norm clip value.
    :param eps: float. Epsilon to prevent division by zero.
    :param exclude_bias: bool. If True, exclude bias parameters from gradient clipping.
    """
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            if exclude_bias and p.ndim == 1:
                # Skip bias terms (typically 1D parameters)
                continue
            # Apply AGC to the parameter and its gradient
            clipped_grad = agc(p, p.grad, agc_eps, agc_clip_val, eps)
            # Replace the original gradient with the clipped gradient
            p.grad.data.copy_(clipped_grad)