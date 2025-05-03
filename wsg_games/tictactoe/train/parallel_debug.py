import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
# Import move_to_and_update_config from the correct location
from transformer_lens.utilities.devices import move_to_and_update_config
import transformer_lens.utils as utils # Import utils
# REMOVED: import transformer_lens.devices as devices
import warnings
import os
import traceback # For detailed error reporting
import datetime # For timestamp
# REMOVED unnecessary type hint imports
# from typing import Union, List, Optional, Tuple, Literal
# from jaxtyping import Int, Float
# REMOVED: from transformer_lens.hook_points import USE_DEFAULT_VALUE


# Optional: Suppress specific warnings
# warnings.filterwarnings("ignore", message="Using String Arguments in *.to is Deprecated")

# Optional: Control which GPUs PyTorch sees
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # Set this BEFORE importing torch if used

# --- Basic Info ---
print(f"Script execution started: {datetime.datetime.now()}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        try:
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"Could not get name for device {i}: {e}")
else:
    print("CUDA is not available, exiting.")
    exit()

# Check if there are at least two GPUs
if torch.cuda.device_count() < 2:
    print("Less than 2 GPUs detected. This script requires at least 2 GPUs (cuda:0 and cuda:1). Exiting.")
    exit()

# --- Define Model Configuration ---
cfg = HookedTransformerConfig(
    n_ctx=1,          # Minimal context
    d_model=1,        # Minimal dimensions
    d_head=1,         # Minimal dimensions
    n_heads=1,        # Minimal heads
    n_layers=1,       # Minimal layers
    d_vocab=1,        # Minimal vocab
    act_fn="relu",    # Standard activation function
    normalization_type="LN", # Explicitly LayerNorm (default)
    positional_embedding_type="standard", # Explicitly standard (default)
    device = None     # Start on CPU, let move_to handle device placement
)
print(f"\n--- Initializing Model ---")
try:
    model = HookedTransformer(cfg)
except Exception as e:
    print(f"Error initializing HookedTransformer: {e}")
    traceback.print_exc()
    exit()

print(f"Initial model device (from config): {model.cfg.device}")
# Check initial parameter device defensively
try:
    initial_param_device = next(model.parameters()).device
    print(f"Initial actual parameter device: {initial_param_device}")
except StopIteration:
    print("Warning: Model has no parameters.")


# --- Check Available Hooks ---
# Print available hooks to help with debugging hook names
print("\n--- Available Hooks in Model ---")
available_hook_names = sorted(list(model.hook_dict.keys()))
for name in available_hook_names:
    print(f"  {name}")
print("-----------------------------")


# --- GPU 0 Setup ---
device0 = torch.device("cuda:0")
print(f"\n--- Moving model to {device0} ---")
try:
    model = move_to_and_update_config(model, device0)
    print(f"Model device after move to {device0} (model.cfg.device): {model.cfg.device}")
    # Verify a parameter's device after move
    if len(list(model.parameters())) > 0:
         print(f"Token embed W_E device after move to {device0}: {model.embed.W_E.device}")
         # Check pos_embed structure before accessing W_pos
         if hasattr(model, 'pos_embed') and hasattr(model.pos_embed, 'W_pos'):
              print(f"Pos embed W_pos device after move to {device0}: {model.pos_embed.W_pos.device}")
         else:
              print("Pos embed type does not have W_pos or pos_embed module not found.")
    else:
         print("Model has no parameters to check.")
except Exception as e:
    print(f"Error moving model to {device0}: {e}")
    traceback.print_exc()
    exit()

# --- GPU 1 Setup ---
device1 = torch.device("cuda:1")
print(f"\n--- Moving model from {model.cfg.device} to {device1} ---")
try:
    model = move_to_and_update_config(model, device1)
    print(f"Model device after move to {device1} (model.cfg.device): {model.cfg.device}")
    # Verify a parameter's device after move
    if len(list(model.parameters())) > 0:
        print(f"Token embed W_E device after move to {device1}: {model.embed.W_E.device}")
        if hasattr(model, 'pos_embed') and hasattr(model.pos_embed, 'W_pos'):
             print(f"Pos embed W_pos device after move to {device1}: {model.pos_embed.W_pos.device}")
        else:
              print("Pos embed type does not have W_pos or pos_embed module not found.")
    else:
        print("Model has no parameters to check.")
except Exception as e:
    print(f"Error moving model to {device1}: {e}")
    traceback.print_exc()
    exit()

# --- Prepare Input ---
x = torch.zeros((1, 1), dtype=torch.long, device=device1)
print(f"\nInput tensor 'x' device: {repr(x.device)}") # Use repr for clarity

# --- Check and SET Default Device ---
current_default_device_idx = torch.cuda.current_device()
print(f"Current default CUDA device index BEFORE setting: {current_default_device_idx}")
print(f"Intended device for inference: {device1}")
print(f"Model expected device (model.cfg.device): {model.cfg.device}") # Note: May show 'cuda'

# Set the default CUDA device to match the target device for inference
if current_default_device_idx != device1.index:
    print(f"--> Setting default CUDA device to: {device1.index}")
    try:
        torch.cuda.set_device(device1) # Use the torch.device object or its index
        current_default_device_idx = torch.cuda.current_device()
        print(f"Current default CUDA device index AFTER setting: {current_default_device_idx}")
    except Exception as e:
        print(f"Error setting default CUDA device to {device1.index}: {e}")
        traceback.print_exc()
        exit()
else:
    print(f"Default CUDA device is already {device1.index}")


# --- Define Hook Function ---
# (This function will be used for all hooks)
def check_tensor_device_hook(tensor: torch.Tensor, hook):
    """Prints the device of the tensor at the specified hook point."""
    # Assume model parameters reflect the true target device
    try:
        # Ensure comparison is with the specific device object (e.g., device(type='cuda', index=1))
        expected_device = next(model.parameters()).device
    except StopIteration:
         expected_device = device1 # Fallback if no parameters

    # Use repr(tensor.device) for unambiguous device output like device(type='cuda', index=1)
    print(f"---> Hook '{hook.name}': \tTensor device = {repr(tensor.device)} | Expected device = {repr(expected_device)}")
    if tensor.device != expected_device:
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"!!!! WARNING: Tensor from {hook.name:<25} is on {repr(tensor.device)} !!!!")
         print(f"!!!! Expected device was {repr(expected_device):<25}              !!!!")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return tensor

# --- Set Up Hooks (Embed/PosEmbed hooks commented out) --- # MODIFIED SECTION
# Define hook points focusing only on block input and first LN
hook_points_to_add = [
    # Embedding Layer Outputs (COMMENTED OUT)
    # "hook_embed",
    # "hook_pos_embed",

    # Input to the first block (KEEP THIS)
    utils.get_act_name("resid_pre", 0),

    # First LayerNorm (KEEP THIS)
    utils.get_act_name("normalized", 0, "ln1"),
]

# Filter list based on hooks actually available in the model's hook_dict
fwd_hooks = []
print("\nSetting up hooks (Embed/PosEmbed hooks *REMOVED*):") # Modified Print statement
for name in hook_points_to_add:
    if name in available_hook_names: # Use the list printed earlier
        fwd_hooks.append((name, check_tensor_device_hook))
        print(f"  - {name}")
    else:
        print(f"  - {name} (Skipped: Not found in model.hook_dict)")


# --- Manual Embedding Addition Test ---
# (KEEP THE MANUAL TEST BLOCK EXACTLY AS IT WAS for comparison)
print("\n--- Manual Embedding Addition Test ---")
# Ensure model is on the correct device before testing
if len(list(model.parameters())) > 0 and next(model.parameters()).device != device1:
    print(f"Warning: Model not on target device {device1} before manual test. Moving...")
    model = move_to_and_update_config(model, device1)

model.eval() # Ensure model is in eval mode if it affects layers like dropout
try:
    with torch.no_grad():
        # Ensure test input x is on the correct device
        test_x = torch.zeros((1, 1), dtype=torch.long, device=device1)
        print(f"Test input 'test_x' device: {repr(test_x.device)}")

        # Manually get embeddings (model should be on device1)
        token_embed_test = model.embed(test_x)
        pos_embed_test = model.pos_embed(test_x)
        print(f"Manual token_embed_test device: {repr(token_embed_test.device)}")
        print(f"Manual pos_embed_test device: {repr(pos_embed_test.device)}")

        # Perform the addition
        sum_embed_test = token_embed_test + pos_embed_test
        print(f"Sum 'sum_embed_test' device: {repr(sum_embed_test.device)}")

        # Explicit check
        if sum_embed_test.device != device1:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! Manual addition test FAILED: Result is on WRONG device !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print("Manual addition test PASSED: Result is on CORRECT device.")

except Exception as e_test:
    print(f"Manual test failed with Exception: {e_test}")
    traceback.print_exc()
print("------------------------------------\n")


# --- Run Inference with Hooks ---
print(f"\n--- Running inference on {device1} (default device set to {torch.cuda.current_device()}) with ONLY resid_pre/ln1 hooks ---") # Modified Print
try:
    with torch.no_grad():
        output = model.run_with_hooks(
            x, # Use the original input 'x' for the actual run
            fwd_hooks=fwd_hooks # Use the modified list without embed hooks
        )
    print("\n------------------------")
    print("Inference successful!")
    print("------------------------")
    print(f"Output value shape: {output.shape}")
    print(f"Output tensor device: {repr(output.device)}")

except RuntimeError as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!! INFERENCE FAILED !!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR MESSAGE: {e}")
    print("\n--- Debugging Device Information at Failure ---")
    print(f"Model expected device (model.cfg.device): {model.cfg.device}")
    try:
        actual_model_device = next(model.parameters()).device
        print(f"Model actual parameter device: {repr(actual_model_device)}")
    except StopIteration:
         print("Model has no parameters to check device.")
    print(f"Input tensor 'x' device: {repr(x.device)}")
    print(f"Default CUDA device index at failure: {torch.cuda.current_device()}")
    # ...(rest of parameter/buffer checking remains the same)...
    print("\nChecking ALL parameter devices:")
    all_params_on_target = True
    param_devices = set()
    if len(list(model.parameters())) > 0:
        for name, param in model.named_parameters():
            param_devices.add(str(param.device))
            is_correct = param.device == device1
            print(f"  {name:<25} {str(param.device):<10} {'(OK)' if is_correct else '<--- WRONG DEVICE'}")
            if not is_correct: all_params_on_target = False
        print(f"All parameters on target device ({device1})? {'YES' if all_params_on_target else 'NO'}")
        print(f"Unique parameter devices found: {param_devices}")
    else:
        print("  Model has no parameters.")

    print("\nChecking ALL buffer devices:")
    all_buffers_on_target = True
    buffer_devices = set()
    if len(list(model.named_buffers())) > 0:
        for name, buf in model.named_buffers():
            buffer_devices.add(str(buf.device))
            is_correct = buf.device == device1
            print(f"  {name:<25} {str(buf.device):<10} {'(OK)' if is_correct else '<--- WRONG DEVICE'}")
            if not is_correct: all_buffers_on_target = False
        print(f"All buffers on target device ({device1})? {'YES' if all_buffers_on_target else 'NO'}")
        print(f"Unique buffer devices found: {buffer_devices}")
    else:
        print("  Model has no buffers.")


except Exception as e:
      print(f"\n!!!!!!!! An unexpected error occurred: {type(e).__name__} !!!!!!!!")
      print(e)
      traceback.print_exc()

print(f"\n--- Script Finished: {datetime.datetime.now()} ---")