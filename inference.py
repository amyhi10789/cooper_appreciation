import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora"
TOKEN = "cooper_person"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_model_cpu_offload()
pipe.load_lora_weights(LORA_PATH)

def build_prompt(user_prompt: str) -> str:
    if "cooper" or "cooper sigrist" in user_prompt.lower():
        return (
            f"{user_prompt}, with {TOKEN} shown in RGB color"
            "make sure the image does not look AI-generated"
        )
    else:
        return (
            f"{user_prompt}, with"
            f"{TOKEN} as a visible figure in the background, easily identifiable as {TOKEN}, all in RGB color"
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
