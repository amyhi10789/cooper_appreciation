from diffusers import DiffusionPipeline
import torch

# Check CUDA availability
assert torch.cuda.is_available(), "CUDA is not available!"

device = "cuda"

# Load SDXL Base 1.0
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,   # required for SDXL on GPU
    variant="fp16",
    use_safetensors=True
)

# Move model to GPU
pipe.to(device)

# Optional but recommended (if xformers is installed)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    print("xFormers not available, continuing without it")

# Prompt and generate image
prompt = "A fantasy castle on a hill at sunset"

image = pipe(
    prompt,
    height=1024,     # safe on A10/A100
    width=1024,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# Save output
image.save("sdxl_output.png")
print("Image saved as sdxl_output.png")
