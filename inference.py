import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-500"
TOKEN = "cooper_person"

NEGATIVE_PROMPT = (
    "child, teenager, female, different person, "
    "cartoon, anime, illustration, painting, oil painting, watercolor, pastel, "
    "storybook, fantasy portrait, vintage portrait, romantic painting, "
    "distorted face, extra facial features, "
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
            f"ultra realistic photo of {TOKEN}, young adult man mid 20s, "
            "DSLR portrait photograph, 85mm lens, "
            "head and shoulders portrait, "
            "round face shape, rounded midface, full cheeks, "
            "gentle jawline, short soft chin, "
            "average-sized straight nose, "
            "medium soft eyebrows, small ears, "
            "trimmed short light brown beard with real hair texture, "
            "short light brown hair mostly covered by a dark blue cap, "
            "relaxed facial muscles, calm neutral expression, candid look, "
            "soft diffused window light, indoor portrait, "
            "natural imperfect skin texture, subtle pores, "
            "slight facial asymmetry, average-looking, "
            "not a painting, not an illustration"
        )

    else:
        return (
            f"ultra realistic photo of {TOKEN}, {user_prompt}, "
            "DSLR photograph, natural human proportions"
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
