import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

def save_batch(images, folder, start_idx, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(folder, f"{prefix}_{start_idx + i:05d}.png"))

# --- CONFIG ---
seed = 42
device = 'cuda:1'
NUM_IMAGES = 128
BATCH_SIZE = 32
CHECKPOINT = "checkpoints/our_face.pt"
PROMPT = "a photo of a person, single person, single face, ultra detailed, raw photo, realistic face"
NEG_PROMPT = "(((deformed))), grayscale, closeup, cartoonish, unrealistic, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fused ears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears"
OUT_DIR = "rebuttal/faces/debiased_3"
# ---------------

pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to(device)

for i in tqdm(range(0, NUM_IMAGES, BATCH_SIZE),desc="Debiased"):
    result, grad_list = pipeline(
        prompt=PROMPT,
        generator=torch.Generator(device).manual_seed(seed + i),
        guidance_scale=7.5,
        num_images_per_prompt=BATCH_SIZE,
        negative_prompt=NEG_PROMPT,
        loss_strength=2500,
        scaling_strength=2600,
        checkpoint_path=CHECKPOINT,
        mode="distribution"
    )
    save_batch(result.images, OUT_DIR, start_idx=i, prefix="debiased")

# import os
# import torch
# from diffusers import DiffusionPipeline
# from tqdm import tqdm

# def save_batch(images, folder, start_idx, prefix):
#     os.makedirs(folder, exist_ok=True)
#     for i, img in enumerate(images):
#         img.save(os.path.join(folder, f"{prefix}_{start_idx + i:05d}.png"))

# # --- CONFIG ---
# seed = 42
# device = 'cuda:1'
# NUM_IMAGES_PER_PROMPT = 100
# BATCH_SIZE = 32
# CHECKPOINT = "pretrained/face.pt"
# OUT_DIR = "new_final_results/gender/debiased_lace"

# PROMPT_LIST = [
#     "A photo of a person"
# ]
# # ---------------

# pipeline = DiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
# ).to(device)

# # --- Loop Over Prompts ---
# for prompt_idx, prompt in enumerate(PROMPT_LIST):
#     prompt_tag = f"prompt{prompt_idx:02d}"
#     prompt_out_dir = os.path.join(OUT_DIR, prompt_tag)
#     os.makedirs(prompt_out_dir, exist_ok=True)

#     # Count already-generated images
#     existing_images = len([
#         f for f in os.listdir(prompt_out_dir)
#         if f.startswith(prompt_tag) and f.endswith(".png")
#     ])
#     print(f"\nGenerating debiased samples for prompt {prompt_idx}: {prompt}")
#     print(f"Already generated: {existing_images} — resuming from there")

#     for i in tqdm(range(existing_images, NUM_IMAGES_PER_PROMPT, BATCH_SIZE), desc=f"Debiased - {prompt_tag}"):
#         curr_batch_size = min(BATCH_SIZE, NUM_IMAGES_PER_PROMPT - i)

#         # This will raise and exit on OOM
#         result, _ = pipeline(
#             prompt=[prompt] * curr_batch_size,
#             generator=torch.Generator(device).manual_seed(seed + i),
#             guidance_scale=7.5,
#             num_images_per_prompt=1,
#             loss_strength=1500,
#             scaling_strength=1600,
#             checkpoint_path=CHECKPOINT,
#             mode="distribution"
#         )

#         save_batch(result.images, prompt_out_dir, start_idx=i, prefix=prompt_tag)
