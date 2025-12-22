from diffusers import DiffusionPipeline
import torch

# Load SDXL Base 1.0
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,   # half precision saves RAM
    variant="fp16",
    use_safetensors=True
)

# Use Apple GPU (MPS)
pipe.to("mps")

# Enable memory-saving options for 16GB RAM
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Prompt and generate image
prompt = "A fantasy castle on a hill at sunset"
image = pipe(prompt, height=768, width=768, num_inference_steps=20).images[0]

# Save output
image.save("sdxl_output.png")
print("Image saved as sdxl_output.png")
