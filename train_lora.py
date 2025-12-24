import os
import itertools
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from accelerate import Accelerator
from torchvision import transforms

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, prompt, resolution=512):
        self.image_dir = image_dir
        self.prompt = prompt
        self.resolution = resolution

        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.GaussianBlur(kernel_size = 3, sigma = (0.1, 0.5)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "prompt": self.prompt,
        }

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    mixed_precision = cfg.get("mixed_precision", "fp16")
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=grad_accum)
    device = accelerator.device

    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )


    noise_scheduler = DDPMScheduler.from_pretrained(cfg["model_name_or_path"], subfolder="scheduler")

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

    rank = int(cfg["rank"])
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(cfg.get("lora_alpha", rank)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    resume_from_lora = cfg.get("resume_from_lora", False)
    resume_lora_path = cfg.get("resume_lora_path", None)

    if resume_from_lora and resume_lora_path and os.path.exists(resume_lora_path):
        print(f"Loading LoRA weights from: {resume_lora_path}")
        unet.load_attn_procs(resume_lora_path)
    else:
        print("No LoRA checkpoint found — training from base model")

    print("Trainable params:", sum(p.numel() for p in unet.parameters() if p.requires_grad))

    print("Trainable tensors:", sum(1 for p in unet.parameters() if p.requires_grad))

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params found. LoRA may not have attached correctly.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        eps=1e-8,
    )

    resolution = int(cfg.get("resolution", 1024))
    dataset = ImageTextDataset(
        cfg["instance_data_dir"],
        cfg["instance_prompt"],
        resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    dataloader, optimizer, unet, vae, text_encoder_1, text_encoder_2 = accelerator.prepare(
        dataloader, optimizer, unet, vae, text_encoder_1, text_encoder_2
    )


    unet.train()
    global_step = 0
    max_train_steps = int(cfg["max_train_steps"])
    log_every = int(cfg.get("log_every", 50))

    for epoch in range(int(cfg.get("num_epochs", 1000))):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device)

                with torch.no_grad():
                    with accelerator.autocast():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor


                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompts = batch["prompt"]
                if isinstance(prompts, str):
                    prompts = [prompts] * bsz

                input_ids_1 = tokenizer_1(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)

                input_ids_2 = tokenizer_2(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)

                with torch.no_grad():
                    enc1 = text_encoder_1(input_ids_1, return_dict=True).last_hidden_state
                    enc2_out = text_encoder_2(input_ids_2, return_dict=True)
                    enc2 = enc2_out.last_hidden_state

                    encoder_hidden_states = torch.cat([enc1, enc2], dim=-1) 
                    pooled_text_embeds = enc2[:, 0, :]         

                if pooled_text_embeds.dim() == 1:
                    pooled_text_embeds = pooled_text_embeds.unsqueeze(0)

                time_ids = torch.tensor(
                    [[resolution, resolution, 0, 0, resolution, resolution]] * bsz,
                    device=device,
                    dtype=pooled_text_embeds.dtype,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_text_embeds,
                        "time_ids": time_ids,
                    },
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction_type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if accelerator.is_main_process and global_step % log_every == 0:
                print(f"Step {global_step}/{max_train_steps} | Loss {loss.item():.4f}")
                ckpt_dir = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)

                unet_to_save = accelerator.unwrap_model(unet)
                unet_to_save.save_lora_adapter(ckpt_dir)

                print(f"Saved LoRA checkpoint to {ckpt_dir}")

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        outdir = cfg["output_dir"]
        os.makedirs(outdir, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)

        unet_to_save.save_attn_procs(outdir)

        with open(os.path.join(outdir, "lora_info.txt"), "w") as f:
            f.write(f"base_model: {cfg['model_name_or_path']}\n")
            f.write(f"rank: {rank}\n")
            f.write(f"learning_rate: {cfg['learning_rate']}\n")
            f.write(f"resolution: {resolution}\n")
            f.write(f"steps: {max_train_steps}\n")
            f.write(f"prompt: {cfg['instance_prompt']}\n")

        print(f"Saved LoRA weights to: {outdir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()