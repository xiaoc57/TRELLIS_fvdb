from safetensors import safe_open
from safetensors.torch import save_file
from einops import rearrange
import torch

def modify_checkpoint(input_path: str, output_path: str):
    # Load the original checkpoint
    with safe_open(input_path, framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    # Create new state dict with modified keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if ".conv.weight" in key:
            new_key = key.replace(".conv.weight", ".weight")
            # BWHDC -> BCDHW
            new_state_dict[new_key] = rearrange(value, "B W H D C -> B C D H W").contiguous()
        elif ".conv.bias" in key:
            new_key = key.replace(".conv.bias", ".bias")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Save the modified checkpoint
    save_file(new_state_dict, output_path)

if __name__ == "__main__":
    input_path = "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"  # Replace with your input path
    output_path = "ckpts/slat_flow_img_dit_L_64l8p2_fp16_modified.safetensors"  # Replace with your output path
    
    modify_checkpoint(input_path, output_path)
    print(f"Checkpoint modified and saved to {output_path}")