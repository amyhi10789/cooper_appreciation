import os
import itertools
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline
from diffusers.models.lora import LoRALinearLayer
from torchvision import transforms


class ImageTextDataset(Dataset):
    def __init__(self, image_dir, prompt, resolution=1024):
        self.image_dir = image_dir
        self.prompt = prompt
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

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

        return {
            "pixel_values": self.transform(image),
            "prompt": self.prompt,
        }


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "fp16")
    )
    device = accelerator.device

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    if torch.cuda.is_available():
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    unet = pipe.unet
    vae = pipe.vae
    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    for p in itertools.chain(
        unet.parameters(),
        vae.parameters(),
        text_encoder_1.parameters(),
        text_encoder_2.parameters(),
    ):
        p.requires_grad_(False)

    lora_layers = []

    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            lora = LoRALinearLayer(
                module.in_features,
                module.out_features,
                rank=cfg["rank"],
            )
            module.lora_layer = lora
            lora_layers.append(lora)

    optimizer = torch.optim.AdamW(
        itertools.chain(*(l.parameters() for l in lora_layers)),
        lr=cfg["learning_rate"],
    )

    dataset = ImageTextDataset(
        cfg["instance_data_dir"],
        cfg["instance_prompt"],
        cfg["resolution"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        num_workers=4,
    )

    dataloader, optimizer, unet = accelerator.prepare(
        dataloader, optimizer, unet
    )

    unet.train()
    global_step = 0

    for epoch in range(999999):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                )

                noisy_latents = pipe.scheduler.add_noise(
                    latents, noise, timesteps
                )

                input_ids_1 = tokenizer_1(
                    batch["prompt"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)

                input_ids_2 = tokenizer_2(
                    batch["prompt"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)

                encoder_hidden_states = text_encoder_1(input_ids_1)[0]
                encoder_hidden_states_2 = text_encoder_2(input_ids_2)[0]

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": encoder_hidden_states_2,
                    },
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process and global_step % 50 == 0:
                print(f"Step {global_step} | Loss {loss.item():.4f}")

            if global_step >= cfg["max_train_steps"]:
                break

        if global_step >= cfg["max_train_steps"]:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        torch.save(
            {f"lora_{i}": l.state_dict() for i, l in enumerate(lora_layers)},
            os.path.join(cfg["output_dir"], "sdxl_lora.pt"),
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
