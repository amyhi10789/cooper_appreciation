import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora_yday"
TOKEN = "cooper_person"

NEGATIVE_PROMPT = (
    "anime, cartoon, illustration, painting, "
    "cgi, 3d render, plastic skin, doll, "
    "perfect face, overly smooth skin, "
    "sharp jawline, model face, beauty lighting, "
    "unreal lighting, fake"
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.enable_model_cpu_offload()
pipe.load_lora_weights(LORA_PATH)

def build_prompt(user_prompt: str) -> str:
    base = (
        f"include {TOKEN}, "
    )

    user_prompt_lower = user_prompt.lower()
    if "cooper" in user_prompt_lower or "cooper sigrist" in user_prompt_lower:
        return f"{base}, realistic colored photo {user_prompt}, a white man"
    else:
        return f"{base}, {user_prompt}, {TOKEN} is clearly visible and identifiable"

if __name__ == "__main__":
    user_prompt = input("Prompt: ")
    final_prompt = build_prompt(user_prompt)

    image = pipe(
        final_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=35,
        guidance_scale=7.0,
        height=1024,
        width=1024
    ).images[0]

    image.save("test2.png")