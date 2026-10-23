"""Joint augmentations for image + keypoints."""

import math
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# Keypoint index order (verified from notebook documentation):
#   0=left_eye_center      1=right_eye_center
#   2=left_eye_inner       3=left_eye_outer      <- both LEFT eye
#   4=right_eye_inner      5=right_eye_outer     <- both RIGHT eye
#   6=left_brow_inner      7=left_brow_outer     <- both LEFT brow
#   8=right_brow_inner     9=right_brow_outer    <- both RIGHT brow
#  10=nose_tip             (midline, only negate x)
#  11=mouth_left          12=mouth_right
#  13=mouth_center_top    14=mouth_center_bottom (midline, only negate x)
#
# Horizontal flip: swap LEFT<->RIGHT pairs, negate all x coords
_FLIP_PAIRS = [(0, 1), (2, 4), (3, 5), (6, 8), (7, 9), (11, 12)]


class JointRandomHorizontalFlip:
    """Flip image and keypoints horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, keypoints):
        # image: (96, 96, 1) numpy uint8
        # keypoints: (15, 2) float tensor in [-1, 1]
        if random.random() >= self.p:
            return image, keypoints

        image = image[:, ::-1, :].copy()

        kp = keypoints.clone()
        kp[:, 0] = -kp[:, 0]          # negate all x coords
        for i, j in _FLIP_PAIRS:
            kp[[i, j]] = kp[[j, i]]   # swap left/right pairs
        return image, kp


class JointRandomRotation:
    """Rotate image and keypoints by a random angle in [-max_angle, max_angle]."""
    def __init__(self, max_angle=15, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, image, keypoints):
        if random.random() >= self.p:
            return image, keypoints

        angle = random.uniform(-self.max_angle, self.max_angle)
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        pil = Image.fromarray(image[:, :, 0], mode='L')
        pil = TF.rotate(pil, angle)
        image = np.array(pil)[:, :, np.newaxis]

        # In image coords (y-down), CCW rotation by θ:
        #   x' = x*cos(θ) + y*sin(θ)
        #   y' = -x*sin(θ) + y*cos(θ)
        kp = keypoints.clone()
        x = kp[:, 0].clone()
        y = kp[:, 1].clone()
        kp[:, 0] = x * cos_a + y * sin_a
        kp[:, 1] = -x * sin_a + y * cos_a
        return image, kp


class JointColorJitter:
    """Randomly adjust brightness and contrast of image (keypoints unchanged)."""
    def __init__(self, brightness=0.3, contrast=0.3, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image, keypoints):
        if random.random() >= self.p:
            return image, keypoints

        pil = Image.fromarray(image[:, :, 0], mode='L')
        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            pil = TF.adjust_brightness(pil, factor)
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            pil = TF.adjust_contrast(pil, factor)
        image = np.array(pil)[:, :, np.newaxis]
        return image, keypoints


class JointRandomScale:
    """Random crop + resize (zoom-in effect). Keypoints scaled accordingly."""
    def __init__(self, scale_range=(0.85, 1.0), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, keypoints):
        if random.random() >= self.p:
            return image, keypoints

        H, W = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        top  = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        cropped = image[top:top+crop_h, left:left+crop_w, :]
        pil = Image.fromarray(cropped[:, :, 0], mode='L')
        pil = pil.resize((W, H), Image.BILINEAR)
        image = np.array(pil)[:, :, np.newaxis]

        # pixel x = (x_norm + 1) * W/2; after crop+resize: new_norm = ((x_px - left) / crop_w) * 2 - 1
        kp = keypoints.clone()
        kp[:, 0] = ((kp[:, 0] + 1) * (W / 2) - left) / (crop_w / 2) - 1
        kp[:, 1] = ((kp[:, 1] + 1) * (H / 2) - top)  / (crop_h / 2) - 1
        kp = kp.clamp(-1, 1)
        return image, kp


class JointGaussianNoise:
    """Add Gaussian noise to image (keypoints unchanged)."""
    def __init__(self, std=0.03, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, image, keypoints):
        if random.random() >= self.p:
            return image, keypoints
        noise = np.random.normal(0, self.std * 255, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, keypoints


class JointCompose:
    """Chain multiple joint transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, keypoints):
        for t in self.transforms:
            image, keypoints = t(image, keypoints)
        return image, keypoints
