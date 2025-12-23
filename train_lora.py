import os
import yaml
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator

from diffusers import StableDiffusionXLPipeline
from diffusers.models.lora import LoRALinearLayer

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class ImageTextDataset(Dataset):
    def __init__(self, image_dir, prompt, resolution=1024):
        self.image_dir = image_dir
        self.prompt = prompt
        self.resolution = resolution
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.image_dir, self.images[idx])
        ).convert("RGB")

        # SDXL ALWAYS expects 1024x1024
        image = image.resize((self.resolution, self.resolution))

        image = torch.tensor(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
             .float()
             .view(self.resolution, self.resolution, 3)
             / 255.0)
        ).permute(2, 0, 1)

        return {
            "pixel_values": image,
            "prompt": self.prompt,
        }

# -------------------------------------------------
# Training
# -------------------------------------------------
def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # Load SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    unet = pipe.unet
    vae = pipe.vae

    # Freeze base model
    for p in itertools.chain(unet.parameters(), vae.parameters()):
        p.requires_grad = False

    # -------------------------------------------------
    # Apply LoRA (correct way)
    # -------------------------------------------------
    lora_layers = []

    for module in unet.modules():
        if isinstance(module, torch.nn.Linear):
            lora = LoRALinearLayer(
                module.in_features,
                module.out_features,
                rank=cfg["rank"],
            ).to(device)

            module.forward = lora.forward
            lora_layers.append(lora)

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    dataset = ImageTextDataset(
        cfg["instance_data_dir"],
        cfg["instance_prompt"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        itertools.chain(*(l.parameters() for l in lora_layers)),
        lr=cfg["learning_rate"],
    )

    dataloader, optimizer = accelerator.prepare(dataloader, optimizer)

    unet.train()
    global_step = 0

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(999999):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device)
                prompts = batch["prompt"]

                # Encode images
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()

                noisy_latents = pipe.scheduler.add_noise(
                    latents, noise, timesteps
                )

                # ---- SDXL CORRECT PROMPT ENCODING ----
                prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
                    prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

                add_time_ids = pipe._get_add_time_ids(
                    original_size=(1024, 1024),
                    crop_coords_top_left=(0, 0),
                    target_size=(1024, 1024),
                    dtype=prompt_embeds.dtype,
                ).to(device)

                # ---- SDXL CORRECT UNET CALL ----
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")

            if global_step >= cfg["max_train_steps"]:
                break

        if global_step >= cfg["max_train_steps"]:
            break

    # -------------------------------------------------
    # Save LoRA
    # -------------------------------------------------
    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        torch.save(
            {f"lora_{i}": l.state_dict() for i, l in enumerate(lora_layers)},
            os.path.join(cfg["output_dir"], "sdxl_lora.safetensors"),
        )

    accelerator.end_training()

# -------------------------------------------------
if __name__ == "__main__":
    main()
