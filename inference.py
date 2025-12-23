import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/pytorch_lora_weights.safetensors"  # Path where LoRA weights were saved
TOKEN = "cooper_person"

# Load base SDXL model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# Load LoRA weights
pipe.unet.load_attn_procs(LORA_PATH)

pipe.enable_model_cpu_offload()

def build_prompt(user_prompt: str) -> str:
    if "cooper" or "cooper sigrist" in user_prompt.lower():
        return (
            f"{user_prompt}, make sure the person asked by the user is {TOKEN}, in color"
            "make sure the image does not look AI-generated"
        )
    else:
        return (
            f"{user_prompt}, with"
            f"{TOKEN} as a visible figure in the background, all in color"
            "make sure the image does not look AI-generated"
        )

if __name__ == "__main__":
    user_prompt = input("Prompt: ")
    final_prompt = build_prompt(user_prompt)

    image = pipe(
        final_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    image.save("result.png")
    print("Saved result.png")
