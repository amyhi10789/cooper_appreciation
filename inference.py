import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-300"
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
pipe.enable_attention_slicing()

def build_prompt(_: str) -> str:
    return f"{TOKEN}, realistic photo"

if __name__ == "__main__":
    input("Press Enter to generate test image...")
    final_prompt = build_prompt("")

    image = pipe(
        final_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=35,
        guidance_scale=5.5,
        height=1024,
        width=1024,
    ).images[0]

    image.save("result_800.png")
