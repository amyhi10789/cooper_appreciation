import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-500"
TOKEN = "cooper_person"

NEGATIVE_PROMPT = (
    "child, teenager, female, different person, "
    "cartoon, anime, large bushy thick mustache, rustic vibe, illustration, painting, "
    "overly smooth skin, plastic skin, airbrushed, wrinkles, sharp jawline"
    "distorted face, asymmetrical eyes, extra facial features, latino man, sharp facial features"
    "text, watermark, noise, black and white, military tough vibe, forehead wrinkles"
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.enable_model_cpu_offload()
pipe.load_lora_weights(LORA_PATH)


def build_prompt(user_prompt: str) -> str:
    user_prompt_lower = user_prompt.lower()

    if "cooper" in user_prompt_lower or "cooper sigrist" in user_prompt_lower:
        return (
            f"photo of {TOKEN}, adult man, lighter skin, head and shoulders portrait, "
            "light skin, short brown hair mostly covered by a dark blue cap, "
            "trimmed beard, round face, soft straight nose, medium eyebrows, small ears"
            "neutral expression, medium resolution, soft facial features"
            "indoor setting, soft natural lighting, blurry skin texture, "
        )
    else:
        return (
            f"photo of {TOKEN}, {user_prompt}, "
            f"{TOKEN} is clearly visible and identifiable"
        )


if __name__ == "__main__":
    user_prompt = input("Prompt: ")
    final_prompt = build_prompt(user_prompt)

    image = pipe(
        final_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=35,
        guidance_scale=7.0,
        height=1024,
        width=1024,
    ).images[0]

    image.save("test2.png")
