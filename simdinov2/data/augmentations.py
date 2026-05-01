# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import logging
from torchvision import transforms
from .transforms import make_normalize_transform
import random

logger = logging.getLogger("dinov2")


def random_posterize(img, min_bits, max_bits):
    bits = random.randint(min_bits, max_bits)
    return transforms.functional.posterize(img, bits)


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        *,
        global_crops_size=224,
        local_crops_size=96,
        # --- new hyperparameters ---
        posterize_min_bits=3,
        posterize_max_bits=5,
        posterize_p=0.5,
        affine_degrees=10,
        affine_translate=0.1,
        affine_p=0.3,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

        logger.info("###################################")
        logger.info("Using modified DINO augmentation parameters:")
        logger.info(f"Posterize: bits {posterize_min_bits}-{posterize_max_bits}, p={posterize_p}")
        logger.info(f"Affine: degrees={affine_degrees}, translate={affine_translate}, p={affine_p}")
        logger.info("###################################")

        # geometric transforms
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # NEW color/structural augmentation block
        color_affine_block = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [
                        transforms.Lambda(lambda img: random_posterize(img, posterize_min_bits, posterize_max_bits))
                    ],
                    p=posterize_p,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=affine_degrees,
                            translate=(affine_translate, affine_translate),
                        )
                    ],
                    p=affine_p,
                ),
            ]
        )

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        # final transform pipelines
        self.global_transfo1 = transforms.Compose([color_affine_block, self.normalize])
        self.global_transfo2 = transforms.Compose([color_affine_block, self.normalize])
        self.local_transfo = transforms.Compose([color_affine_block, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops
        im1 = self.geometric_augmentation_global(image)
        im2 = self.geometric_augmentation_global(image)

        output["global_crops"] = [self.global_transfo1(im1), self.global_transfo2(im2)]
        output["global_crops_teacher"] = output["global_crops"]

        # local crops
        output["local_crops"] = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]

        output["offsets"] = ()
        return output
