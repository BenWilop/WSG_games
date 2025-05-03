import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utilities.devices import move_to_and_update_config

cfg = HookedTransformerConfig(n_layers=1, d_model=1, n_ctx=1, d_head=1, n_heads=1, d_vocab=1, act_fn="relu")
model = HookedTransformer(cfg)

# GPU 0
torch.cuda.set_device(0)
device0 = torch.device("cuda:0")
model = model.to(device0)

# GPU 1
torch.cuda.set_device(1)
device1 = torch.device("cuda:1")
model = move_to_and_update_config(model, device1)
x = torch.zeros((1, 1), dtype=torch.long, device=device1)
print(model(x))

#   File "/homes/55/bwilop/wsg/WSG_games/uv_venv2/lib/python3.10/site-packages/transformer_lens/components/layer_norm.py", line 56, in forward
#     return self.hook_normalized(x * self.w + self.b).to(self.cfg.dtype)
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!