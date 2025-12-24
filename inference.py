import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-800"
TOKEN = "cooper_person"

NEGATIVE_PROMPT = (
    "anime, cartoon, illustration, painting, "
    "cgi, 3d render, plastic skin, doll, "
    "overly smooth skin, unreal lighting, fake"
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights(LORA_PATH)

def build_prompt(user_prompt: str) -> str:
    p = user_prompt.strip().lower()

    if TOKEN.lower() in p or "cooper" in p:
        return f"{TOKEN}, realistic photo, {user_prompt}"

    return (
        f"{user_prompt}, "
        f"{TOKEN} in the scene, visible in the background, "
        f"off-center, natural presence, realistic photo"
    )


if __name__ == "__main__":
    user_prompt = input("Prompt: ")
    final_prompt = build_prompt(user_prompt)

    image = pipe(
        final_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=35,
        guidance_scale=5.5,
        height=1024,
        width=1024,
    ).images[0]

    image.save("result.png")
