# Z-Image Transformer Setup for ComfyUI

**Note:** The Z-Image transformer is FP32. This method uses a modified script from ComfyUI with QKV mappings. It does **not** modify `supported_models.py`.
*FP32 may be overkill for some setups.*

## Prerequisites
*   **System RAM:** Ensure you have at least **32GB of System RAM**, as you will be processing a massive 24GB FP32 model.

---

## Method 1: Convert and Merge (Recommended)

### 1. Download Required Files
*   **Download the script:**
    [Z-Image_convert_and_merge.py](https://github.com/PixWizardry/ComfyUI_Z-Image_FP32/blob/main/Z-Image_convert_and_merge.py)
*   **Download the transformer files:**
    [Z-Image-Turbo Transformer Files](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/transformer)

### 2. Prepare the Folder
Place the downloaded python script and the transformer files (safe tensors) into the **same folder**.

### 3. Run the Merge Command
Open your command line interface (CMD) from that directory and run the following command:

```bash
python Z-Image_convert_and_merge.py Z-Image_FP32.safetensors diffusion_pytorch_model-00001-of-00003.safetensors diffusion_pytorch_model-00002-of-00003.safetensors diffusion_pytorch_model-00003-of-00003.safetensors
```

> **CRITICAL:** Do not include "bf16" or "fp16" in the output filename (`Z-Image_FP32.safetensors`). If you do, the script will attempt to convert it. We want it to default to **FP32**.

### 4. Load in ComfyUI
1.  Refresh ComfyUI.
2.  Use the **"Diffusion Model Loader KJ"** custom node.
3.  Set the settings to **FP32**.
4.  Load the generated safetensor and enjoy.

---

## Method 2: Legacy Method (Modifies Core Files)

> **WARNING:** THIS IS A BETA MOD. DO IT AT YOUR OWN RISK.
> This mod is designed for ComfyUI **Version 3.75**. It may fail on newer or older versions.

### Instructions
1.  **Backup:** Create a backup of your original `supported_models.py` located in `ComfyUI\comfy`.
2.  **Install Mod:** Drag and drop the modded version to that folder to support the FP32 Z-Image model.
3.  **Merge Files:** Go to [HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and merge their Transformer files into 1 file. *(Requires at least 32GB of CPU RAM to merge)*.
4.  **Run:** Load your Safetensor and enjoy.

### Common Errors
*   `unet missing: ['norm_final.weight']`
    *   *Note:* The image will usually still generate despite this error.
