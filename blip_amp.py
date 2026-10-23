"""Apply an AMP optimizer update without committing nonfinite gradients."""
import torch


def amp_optimizer_step(optimizer, scaler, parameters, max_norm=1.):
    parameters = list(parameters)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm).item():
        # A finite forward loss can still overflow during FP16 backward.
        # Back off the scale and discard this update, preserving the weights.
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        optimizer.zero_grad(set_to_none=True)
        return False, float(norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True, float(norm)
