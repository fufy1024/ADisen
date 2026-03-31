import random
from pathlib import Path
from typing import List, Optional

import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop


class PromptDataset(Dataset):
    """A simple dataset to prepare prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt: str, num_samples: int):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


class DiffusionDataset(Dataset):
    """
    SDXL dataset version of DiffusionDataset.

    Main differences from the SD1.x version:
    - no longer tokenizes prompt in the dataset
    - returns raw prompt strings because SDXL needs two tokenizers/two text encoders
    - returns micro-conditioning metadata: original_size and crop_top_left
    """

    def __init__(
        self,
        concepts_list,
        size: int = 1024,
        center_crop: bool = False,
        with_prior_preservation: bool = False,
        num_class_images: int = 200,
        hflip: bool = False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.with_prior_preservation = with_prior_preservation
        self.num_class_images = num_class_images
        self.hflip = hflip
        self.interpolation = PIL.Image.BILINEAR

        if len(concepts_list) != 1:
            raise ValueError("This SDXL port currently expects one concept entry.")

        concept = concepts_list[0]
        self.instance_prompt = concept["instance_prompt"]
        self.class_prompt = concept.get("class_prompt")
        self.instance_images_path = self._gather_images(concept["instance_data_dir"])

        if not self.instance_images_path:
            raise ValueError(f"No training images found in {concept['instance_data_dir']}")

        self.class_images_path: List[Path] = []
        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.image_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images_available = len(self.class_images_path)
        self._length = max(self.num_instance_images, self.num_class_images_available) if with_prior_preservation else self.num_instance_images

    @staticmethod
    def _gather_images(path_str: str) -> List[Path]:
        path = Path(path_str)
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted([p for p in path.iterdir() if p.is_file()])
        return []

    def __len__(self):
        return self._length

    def _process_image(self, image: Image.Image):
        image = image.convert("RGB")
        original_size = (image.height, image.width)
        image = self.train_resize(image)
        if self.hflip and random.random() < 0.5:
            image = self.train_flip(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, (self.size, self.size))
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        pixel_values = self.image_transforms(image)
        return pixel_values, original_size, crop_top_left

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        pixel_values, original_size, crop_top_left = self._process_image(instance_image)
        example["instance_images"] = pixel_values
        example["instance_prompt"] = self.instance_prompt
        example["instance_original_size"] = original_size
        example["instance_crop_top_left"] = crop_top_left
        
        if self.with_prior_preservation:
            class_image = Image.open(self.class_images_path[index % self.num_class_images_available])
            class_pixel_values, class_original_size, class_crop_top_left = self._process_image(class_image)
            example["class_images"] = class_pixel_values
            example["class_prompt"] = self.class_prompt
            example["class_original_size"] = class_original_size
            example["class_crop_top_left"] = class_crop_top_left

        return example


def collate_fn(examples, with_prior_preservation: bool = False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["instance_original_size"] for example in examples]
    crop_top_lefts = [example["instance_crop_top_left"] for example in examples]

    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        original_sizes += [example["class_original_size"] for example in examples]
        crop_top_lefts += [example["class_crop_top_left"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }



# =========================
# SDXL prompt encoding utils
# =========================
def encode_prompt_sdxl_for_ddim(
    prompt,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device,
    out_dtype,
):
    prompt_2 = prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        prompt_2,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    text_input_ids_2 = text_inputs_2.input_ids.to(device)

    enc_out_1 = text_encoder(
        text_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    prompt_embeds_1 = enc_out_1.hidden_states[-2]

    enc_out_2 = text_encoder_2(
        text_input_ids_2,
        output_hidden_states=True,
        return_dict=True,
    )
    prompt_embeds_2 = enc_out_2.hidden_states[-2]

    pooled_prompt_embeds = enc_out_2.text_embeds

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    prompt_embeds = prompt_embeds.to(device=device, dtype=out_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=out_dtype)

    return prompt_embeds, pooled_prompt_embeds


def make_add_time_ids_sdxl(
    unet,
    text_encoder_2,
    original_size,
    crops_coords_top_left,
    target_size,
    batch_size,
    device,
    dtype,
):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        unet.config.addition_time_embed_dim * len(add_time_ids)
        + text_encoder_2.config.projection_dim
    )
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"SDXL add_time_ids dim mismatch: expected {expected_add_embed_dim}, got {passed_add_embed_dim}"
        )

    add_time_ids = torch.tensor([add_time_ids], device=device, dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

