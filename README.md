# ComfyUI_Z-Image_FP32
The transformer is FP32. ComfyUI's script modify for FP32 with the QKV mappings instead of my other MOD. FP32 could be overkill.
This Method does not modify "supported_models.py".

Download code: https://github.com/PixWizardry/ComfyUI_Z-Image_FP32/blob/main/Z-Image_convert_and_merge.py

Download transformer files: (RAM: Ensure you have at least 32GB of System RAM, since you are processing the massive 24GB FP32 model.)
https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/transformer

Put everything in the Same folder and open the commandline from the same directory, where you saved everything.

CMD to merge:  
python Z-Image_convert_and_merge.py Z-Image_FP32.safetensors diffusion_pytorch_model-00001-of-00003.safetensors diffusion_pytorch_model-00002-of-00003.safetensors diffusion_pytorch_model-00003-of-00003.safetensors
Crucial: Do not put "bf16" or "fp16" in the output filename, otherwise the script will convert it. We want it to default to FP32.

Refresh ComfyUI
Use "Diffusion Model Loader KJ" custom node, and set the settings to FP32, to load the safetensor and Enjoy.

Old Method:
Mod is for ComfyUI ***VERSION 3.75***. You could try it on new or older version but be warn, it could fail.

THIS IS A BETA MOD, DO AT YOUR OWN RISK.

Backup your orginal "supported_models.py" at "ComfyUI\comfy"
Drag and drop this modded version to support FP32 Z-Image model

THIS IS A BETA MOD, DO AT YOUR OWN RISK.

Common Error: "unet missing: ['norm_final.weight']"
Image will still generate.

Goto https://huggingface.co/Tongyi-MAI/Z-Image-Turbo 
Merge their Transformer files into 1 file. (You need at Least 32GB of CPU Ram to merge)

Load your Safetensor and Enjoy.
