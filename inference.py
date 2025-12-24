import torch
from diffusers import StableDiffusionXLPipeline

LORA_PATH = "output/cooper_lora/checkpoint-1000"
TOKEN = "cooper_person"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# IMPORTANT: no adapter_name, no set_adapters
pipe.load_lora_weights(LORA_PATH)

generator = torch.Generator("cuda").manual_seed(1234)

prompt = (
    f"{TOKEN}, ultra realistic photo, DSLR photography, "
    "sharp focus, natural lighting, full color, photorealistic"
)

negative_prompt = (
    "sketch, drawing, illustration, painting, line art, "
    "paper texture, text, watermark, noise"
)

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=35,
    guidance_scale=4.5,
    height=1024,
    width=1024,
    cross_attention_kwargs={"scale": 1.0},
).images[0]

image.save("result_1000.png")
