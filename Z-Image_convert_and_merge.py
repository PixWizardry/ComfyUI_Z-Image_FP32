import safetensors.torch
import torch
import sys
import os

# Usage: python convert_and_merge.py output_filename.safetensors input_shard1.safetensors input_shard2.safetensors ...

if len(sys.argv) < 3:
    print("Usage: python convert_and_merge.py <output_file> <input_file1> [input_file2 ...]")
    sys.exit(1)

output_file = sys.argv[1]
input_files = sys.argv[2:]

# --- 1. Determine Precision based on Filename ---
# If filename doesn't say "bf16" or "fp16", it defaults to FP32 (Original)
cast_to = None
if "fp8" in output_file:
    cast_to = torch.float8_e4m3fn
    print("Format detected: FP8")
elif "fp16" in output_file:
    cast_to = torch.float16
    print("Format detected: FP16")
elif "bf16" in output_file:
    cast_to = torch.bfloat16
    print("Format detected: BF16")
else:
    print("No format tag found in filename. Keeping original precision (FP32).")

# Key replacements to make it ComfyUI Native
replace_keys = {
    "all_final_layer.2-1.": "final_layer.",
    "all_x_embedder.2-1.": "x_embedder.",
    ".attention.to_out.0.bias": ".attention.out.bias",
    ".attention.norm_k.weight": ".attention.k_norm.weight",
    ".attention.norm_q.weight": ".attention.q_norm.weight",
    ".attention.to_out.0.weight": ".attention.out.weight"
}

out_sd = {}
cc = {} # Buffer for QKV merging

# --- 2. Load and Process Shards ---
for f in input_files:
    print(f"Loading {os.path.basename(f)}...")
    with safetensors.torch.safe_open(f, framework="pt") as f_open:
        for k in f_open.keys():
            w = f_open.get_tensor(k)
            
            # Cast if requested
            if cast_to is not None:
                w = w.to(cast_to)
                
            k_out = k
            
            # Apply name fixes
            for r, rr in replace_keys.items():
                k_out = k_out.replace(r, rr)

            # Handle QKV combination logic
            if "attention.to_k.weight" in k_out or "attention.to_q.weight" in k_out or "attention.to_v.weight" in k_out:
                base_key = k_out.replace("to_k.weight", "").replace("to_q.weight", "").replace("to_v.weight", "")
                
                if base_key not in cc:
                    cc[base_key] = {}
                
                if "to_q" in k_out: cc[base_key]['q'] = w
                if "to_k" in k_out: cc[base_key]['k'] = w
                if "to_v" in k_out: cc[base_key]['v'] = w
                continue

            if k_out.endswith(".attention.to_out.0.bias"):
                continue
                
            out_sd[k_out] = w

# --- 3. Merge QKV Tensors ---
print("Merging Attention (QKV) layers...")
for base_key, parts in cc.items():
    if 'q' in parts and 'k' in parts and 'v' in parts:
        # Concatenate Q, K, V
        qkv_weight = torch.cat([parts['q'], parts['k'], parts['v']], dim=0)
        out_key = base_key + "qkv.weight"
        out_sd[out_key] = qkv_weight
    else:
        print(f"Warning: Incomplete QKV for {base_key}")

# --- 4. Inject Missing 'norm_final' ---
# This fixes the "unet missing: norm_final.weight" error
if "norm_final.weight" not in out_sd and "x_embedder.weight" in out_sd:
    print("Injecting dummy norm_final.weight for compatibility...")
    ref = out_sd["x_embedder.weight"]
    out_sd["norm_final.weight"] = torch.ones((ref.shape[0],), dtype=ref.dtype, device=ref.device)

# --- 5. Save ---
print(f"Saving merged model to {output_file}...")
safetensors.torch.save_file(out_sd, output_file)
print("Merge Complete!")
