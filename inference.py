import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora"
TOKEN = "cooper_person"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.enable_model_cpu_offload()

pipe.load_lora_weights(LORA_PATH)

def build_prompt(user_prompt: str) -> str:
    base = (
        f"a high quality photo of {TOKEN}, "
        "a real person, natural skin texture, "
        "photorealistic, candid, unposed, "
        "35mm photography, realistic lighting"
    )
    user_prompt_lower = user_prompt.lower()
    if "cooper" in user_prompt_lower or "cooper sigrist" in user_prompt_lower:
        return f"{base}, {user_prompt}"
    else:
        return f"{base}, {user_prompt}, {TOKEN} is clearly visible and identifiable"
    
if __name__ == "__main__":
    user_prompt = input("Prompt: ")
    final_prompt = build_prompt(user_prompt)
    image = pipe(
        final_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    image.save("result.png")
