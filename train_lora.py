"""
train_lora_personX.py
----------------------------------
Trains a LoRA on SDXL to represent Person X.
Author: You
"""

import os
import torch
import itertools
import yaml
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.models.lora import LoRALinearLayer

# ==========================
# Dataset class
# ==========================
class ImageTextDataset(Dataset):
    def __init__(self, image_dir, prompt, resolution=768):
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
        image = image.resize((self.resolution, self.resolution))

        # Convert to PyTorch tensor [C,H,W] and scale to [0,1]
        image = torch.tensor(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
             .float()
             .view(self.resolution, self.resolution, 3) / 255.0)
        ).permute(2, 0, 1)

        return {"pixel_values": image, "prompt": self.prompt}

# ==========================
# Main training function
# ==========================
def main():
    # --------------------------
    # Load config
    # --------------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------
    # Load SDXL base model
    # --------------------------
    print("Loading SDXL base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    ).to(device)

    # Freeze all model parameters (only train LoRA layers)
    for p in itertools.chain(pipe.unet.parameters(), pipe.vae.parameters(), pipe.text_encoder.parameters()):
        p.requires_grad = False

    # --------------------------
    # Replace Linear layers with LoRA
    # --------------------------
    print("Adding LoRA layers...")
    lora_layers = []
    for name, module in pipe.unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            lora = LoRALinearLayer(module.in_features, module.out_features, rank=cfg["rank"]).to(device)
            module.forward = lora(module)
            lora_layers.append(lora)

    # --------------------------
    # Dataset and DataLoader
    # --------------------------
    print("Preparing dataset...")
    dataset = ImageTextDataset(cfg["instance_data_dir"], cfg["instance_prompt"], cfg["resolution"])
    dataloader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True)

    # --------------------------
    # Optimizer
    # --------------------------
    optimizer = torch.optim.AdamW(itertools.chain(*(l.parameters() for l in lora_layers)),
                                  lr=cfg["learning_rate"])

    # --------------------------
    # Training loop
    # --------------------------
    print("Starting training...")
    global_step = 0
    max_steps = cfg["max_train_steps"]

    while global_step < max_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            prompts = batch["prompt"]

            # Encode images into latents
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            # Add noise for diffusion training
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Encode text
            input_ids = pipe.tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states = pipe.text_encoder(input_ids)[0]

            # Predict noise with UNet
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 50 == 0:
                print(f"[Step {global_step}/{max_steps}] Loss: {loss.item():.4f}")

            if global_step >= max_steps:
                break

    # --------------------------
    # Save LoRA weights
    # --------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    save_path = os.path.join(cfg["output_dir"], "pytorch_lora_weights.safetensors")
    torch.save({f"lora_{i}": layer.state_dict() for i, layer in enumerate(lora_layers)}, save_path)
    print(f"Training complete. LoRA weights saved to {save_path}")

# ==========================
# Entry point
# ==========================
if __name__ == "__main__":
    main()
