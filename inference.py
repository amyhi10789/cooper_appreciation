import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-500"
TOKEN = "cooper_person"

NEGATIVE_PROMPT = (
    "child, teenager, female, different person, "
    "cartoon, anime, illustration, painting, "
    "distorted face, asymmetrical eyes, extra facial features, "
    "text, watermark, noise, black and white, "
    "military tough vibe, aggressive expression"
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
            f"photo of {TOKEN}, young adult man in his mid 20s, "
            "head and shoulders portrait, round face shape, "
            "gentle jawline, soft chin, rounded cheeks, "
            "soft straight nose, medium soft eyebrows, small ears, "
            "trimmed short light brown beard, "
            "short light brown hair mostly covered by a dark blue cap, "
            "relaxed facial muscles, calm neutral expression, "
            "soft diffused window light, portrait photography, "
            "shallow depth of field, natural skin texture"
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
