# Libs >>>
import torch
from torch.cuda import Stream 

# Main >>>
def apply_gradient_modifiers(
    model: torch.nn.Module,
    modifiers_fn: list, 
    exclude_layer_types: list = None,
    param_to_stream_ratio: int = 96,
) -> None:
    """
    Asynchronous gradient modification system supporting both CPU and GPU execution.

    Args:
        model (torch.nn.Module): Target PyTorch model whose gradients will be modified.
        modifiers_fn (list): List of modifier configurations, where each configuration is:
            [function, param_first, *args]
            - function: The modifier function to apply
            - param_first: Boolean indicating if parameter should be first argument
            - *args: Additional arguments passed to the modifier function
        exclude_layer_types (list, optional): List of layer type names to skip during processing.
            Example: ['Conv2d', 'Linear']
        param_to_stream_ratio (int, optional): Number of parameters to assign to a single CUDA stream.
            Default is 96. Higher values reduce the number of streams.

    Example:
        >>> modifiers = [
        ...     [normalize_gradient, False],
        ...     [clip_gradient, True, 1.0]
        ... ]
        >>> apply_gradient_modifiers(model, modifiers, ['BatchNorm2d'], param_to_stream_ratio=2)
    """
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"

    # Map module types for efficient layer filtering
    module_types = {
        name: module.__class__.__name__ for name, module in model.named_modules()
    }

    def process_gradient(param, grad):
        """Helper function to process a single gradient"""
        if grad.norm() == 0:
            return None
            
        modified_grad = grad
        for modifier, param_first, *args in modifiers_fn:
            try:
                modified = (
                    modifier(param, modified_grad, *args)
                    if param_first
                    else modifier(modified_grad, *args)
                )
                if modified is not None:
                    modified_grad = modified
            except:
                continue
        return modified_grad

    with torch.no_grad():
        # Collect valid parameters requiring gradient updates
        param_info = [
            (name, param, ".".join(name.split(".")[:-1]))
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        ]

        if is_cuda:
            # Calculate number of streams based on param_to_stream_ratio
            num_streams = (len(param_info) + param_to_stream_ratio - 1) // param_to_stream_ratio
            streams = [Stream() for _ in range(num_streams)]
            
            for i, (_, param, module_name) in enumerate(param_info):
                if exclude_layer_types and module_types.get(module_name) in exclude_layer_types:
                    continue
                
                # Assign parameters to streams in a round-robin fashion
                stream_idx = i // param_to_stream_ratio
                with torch.cuda.stream(streams[stream_idx]):
                    grad = param.grad.detach().clone(memory_format=torch.preserve_format)
                    modified_grad = process_gradient(param, grad)
                    if modified_grad is not None:
                        param.grad.copy_(modified_grad, non_blocking=True)
                        
            torch.cuda.synchronize()
            
        else:
            # Sequential processing for CPU
            for _, param, module_name in param_info:
                if exclude_layer_types and module_types.get(module_name) in exclude_layer_types:
                    continue
                    
                grad = param.grad.detach().clone(memory_format=torch.preserve_format)
                modified_grad = process_gradient(param, grad)
                if modified_grad is not None:
                    param.grad.copy_(modified_grad)