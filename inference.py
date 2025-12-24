import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-1400"
TOKEN = "cooper_person"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights(LORA_PATH)

generator = torch.Generator("cuda").manual_seed(1234)

prompt = (
    f"realistic photo of {TOKEN}, head and shoulders portrait, "
    "neutral expression, facing camera"
)

negative_prompt = (
    "child, teenager, young boy, female, soft face, "
    "sketch, drawing, illustration, painting, line art"
    "paper texture, text, watermark, waxy, glossy, smooth skin"
)

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=35,
    guidance_scale=4.0,
    height=1024,
    width=1024,
).images[0]

image.save("identity_check_1400.png")
