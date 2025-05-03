import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utilities.devices import move_to_and_update_config
import transformer_lens.utils as utils

cfg = HookedTransformerConfig(n_layers=1, d_model=1, n_ctx=1, d_head=1, n_heads=1, d_vocab=1, act_fn="relu")
model = HookedTransformer(cfg)  # <-- cuda:0 (default)

device1 = torch.device("cuda:1")
model = move_to_and_update_config(model, device1)  # <-- should now be on cuda:1
x = torch.zeros((1, 1), dtype=torch.long, device=device1)

def print_device_hook(activation, hook):
    print(f"Hook '{hook.name}': Device={repr(activation.device)}")
    return activation

fwd_hooks = [
    (utils.get_act_name("embed"), print_device_hook),
    (utils.get_act_name("pos_embed"), print_device_hook),
    (utils.get_act_name("resid_pre", 0), print_device_hook),
]
model.run_with_hooks(x, fwd_hooks=fwd_hooks)

# Output:
# Moving model to device:  cuda
# Hook 'hook_embed': Device=device(type='cuda', index=1)
# Hook 'hook_pos_embed': Device=device(type='cuda', index=1)
# Hook 'blocks.0.hook_resid_pre': Device=device(type='cuda', index=0)  # <-- Should be cuda:1 instead
# Traceback (most recent call last):
# ...
# model.run_with_hooks(x, fwd_hooks=fwd_hooks)
# ...
#   File ".../lib/python3.10/site-packages/transformer_lens/components/layer_norm.py", line 56, in forward
#     return self.hook_normalized(x * self.w + self.b).to(self.cfg.dtype)
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
