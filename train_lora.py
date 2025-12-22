import os
import itertools
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator

from diffusers import StableDiffusionXLPipeline
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTokenizer, CLIPTextModel
from peft import get_peft_model, LoraConfig, TaskType

# -------------------------------
# Dataset
# -------------------------------
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
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # Resize to SDXL resolution
        image = image.resize((self.resolution, self.resolution))

        # Convert to tensor
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

# -------------------------------
# Main training loop
# -------------------------------
def main():
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "fp16"))
    device = accelerator.device

    # Load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.float16
    )
    pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Freeze original model
    for p in itertools.chain(unet.parameters(), vae.parameters(), text_encoder.parameters()):
        p.requires_grad = False

    # -------------------------------
    # Apply LoRA
    # -------------------------------
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            lora = LoRALinearLayer(module.in_features, module.out_features, rank=cfg["rank"]).to(device)
            module.forward = lora.forward  # ✅ correct LoRA attachment

    # -------------------------------
    # Dataset and DataLoader
    # -------------------------------
    dataset = ImageTextDataset(
        image_dir=cfg["instance_data_dir"],
        prompt=cfg["instance_prompt"],
        resolution=1024  # SDXL requires 1024x1024
    )
    dataloader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True)

    # Optimizer
    lora_layers = [m for m in unet.modules() if isinstance(m, LoRALinearLayer)]
    optimizer = torch.optim.AdamW(itertools.chain(*(l.parameters() for l in lora_layers)), lr=cfg["learning_rate"])

    # Prepare accelerator
    dataloader, optimizer = accelerator.prepare(dataloader, optimizer)
    unet.train()

    global_step = 0
    max_steps = cfg["max_train_steps"]

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(999999):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device)
                prompts = batch["prompt"]

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                ).long()

                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                input_ids = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    # -------------------------------
    # Save LoRA weights
    # -------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        torch.save(
            {f"lora_{i}": layer.state_dict() for i, layer in enumerate(lora_layers)},
            os.path.join(cfg["output_dir"], "pytorch_lora_weights.safetensors")
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
