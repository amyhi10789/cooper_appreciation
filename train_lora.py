import os
import yaml
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator

from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.models.lora import LoraConfig


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

    # -------------------------------------------------
    # Load SDXL
    # -------------------------------------------------
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # -------------------------------------------------
    # Add LoRA (CORRECT WAY)
    # -------------------------------------------------
    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["rank"],
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="UNET",
    )

    pipe.unet.add_adapter(lora_config)

    # Train only LoRA params
    trainable_params = [
        p for p in pipe.unet.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["learning_rate"],
    )

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

    dataloader, optimizer = accelerator.prepare(dataloader, optimizer)

    pipe.unet.train()
    global_step = 0

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(999999):
        for batch in dataloader:
            with accelerator.accumulate(pipe.unet):
                pixel_values = batch["pixel_values"].to(device)
                prompts = batch["prompt"]

                # Encode images → latents
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

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

                # -------- SDXL PROMPT ENCODING (CORRECT) --------
                prompt_embeds, out2 = pipe.encode_prompt(
                    prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

                pooled_text_embeds = out2.last_hidden_state.mean(dim=1)


                add_time_ids = pipe._get_add_time_ids(
                    original_size=(1024, 1024),
                    crop_coords_top_left=(0, 0),
                    target_size=(1024, 1024),
                    dtype=prompt_embeds.dtype,
                )
                add_time_ids = add_time_ids.repeat(
                    latents.shape[0], 1
                ).to(device)

                # -------- UNET --------
                noise_pred = pipe.unet(
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
        pipe.unet.save_attn_procs(cfg["output_dir"])

    accelerator.end_training()


# -------------------------------------------------
if __name__ == "__main__":
    main()
