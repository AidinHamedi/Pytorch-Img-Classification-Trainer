# Main >>>
def apply_gradient_modifier(model, modifier_fn, *args, **kwargs):
    """
    Applies a gradient modifier function to all trainable parameters in the model.

    This function iterates over all parameters of the model that require gradients
    and have computed gradients, and applies the provided modifier function to each
    gradient tensor. The modified gradients are then assigned back to the parameters.

    It is intended to be used after the backward pass (loss.backward()) and before
    the optimization step (optimizer.step()), typically within the training loop.

    Args:
        model (nn.Module): The model whose gradients will be modified.
        modifier_fn (callable): A function that takes a gradient tensor and returns
            a modified gradient tensor or modifies the tensor in place. The function
            can also accept additional arguments.
        *args: Additional positional arguments to pass to modifier_fn.
        **kwargs: Additional keyword arguments to pass to modifier_fn.
    """
    if not callable(modifier_fn):
        raise TypeError("modifier_fn must be a callable function.")

    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            # Apply the modifier function
            modified_grad = modifier_fn(param.grad, *args, **kwargs)

            # If modifier_fn returns a new gradient, assign it
            if modified_grad is not None:
                param.grad = modified_grad
