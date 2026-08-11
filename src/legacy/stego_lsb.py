"""Legacy custom LSB implementation retained for educational comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STEGO_METHODS = {"lsb", "random-lsb", "edge-lsb"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: str | Path, size: int | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if size is not None:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Could not write image: {path}")


def validate_payload_bits(payload_bits: int) -> None:
    if payload_bits < 1 or payload_bits > 4:
        raise ValueError("payload_bits must be between 1 and 4.")


def validate_method(method: str) -> None:
    if method not in STEGO_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(STEGO_METHODS))}.")


def validate_payload_rate(payload_rate: float) -> None:
    if payload_rate <= 0 or payload_rate > 1:
        raise ValueError("payload_rate must be in the interval (0, 1].")


def hide_array(
    cover_image: np.ndarray,
    secret_image: np.ndarray,
    payload_bits: int = 4,
    method: str = "lsb",
    payload_rate: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Hide secret image high bits inside cover image low bits."""
    validate_payload_bits(payload_bits)
    validate_method(method)
    validate_payload_rate(payload_rate)
    if cover_image.shape != secret_image.shape:
        height, width = cover_image.shape[:2]
        secret_image = cv2.resize(secret_image, (width, height), interpolation=cv2.INTER_AREA)

    cover_pixels = cover_image.astype(np.uint16)
    secret_pixels = secret_image.astype(np.uint16)
    clear_mask = (0xFF >> payload_bits) << payload_bits
    payload_mask = (1 << payload_bits) - 1
    secret_high_bits = secret_pixels >> (8 - payload_bits)

    if method == "lsb" and payload_rate == 1.0:
        cover_high_bits = cover_pixels & clear_mask
        return (cover_high_bits | secret_high_bits).astype(np.uint8)

    stego_pixels = cover_pixels.copy()
    positions = select_embedding_positions(
        cover_image=cover_image,
        payload_rate=payload_rate,
        method=method,
        seed=seed,
    )
    flat_cover = stego_pixels.reshape(-1)
    flat_secret = secret_high_bits.reshape(-1)
    selected_cover = flat_cover[positions]
    selected_secret = flat_secret[positions] & payload_mask
    flat_cover[positions] = (selected_cover & clear_mask) | selected_secret
    return stego_pixels.astype(np.uint8)


def select_embedding_positions(
    cover_image: np.ndarray,
    payload_rate: float,
    method: str,
    seed: int | None,
) -> np.ndarray:
    total_values = cover_image.size
    count = max(1, int(total_values * payload_rate))
    rng = np.random.default_rng(seed)

    if method == "random-lsb":
        return rng.choice(total_values, size=count, replace=False)

    if method == "edge-lsb":
        gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        strength = cv2.magnitude(grad_x, grad_y).reshape(-1)
        pixel_count = cover_image.shape[0] * cover_image.shape[1]
        selected_pixels = np.argsort(strength)[-max(1, min(pixel_count, count // 3)) :]
        channel_offsets = np.array([0, 1, 2], dtype=np.int64)
        positions = (selected_pixels[:, None] * 3 + channel_offsets[None, :]).reshape(-1)
        if len(positions) > count:
            positions = rng.choice(positions, size=count, replace=False)
        return positions

    return np.arange(total_values)


def extract_array(stego_image: np.ndarray, payload_bits: int = 4) -> np.ndarray:
    """Recover the hidden image approximation from the low payload bits."""
    validate_payload_bits(payload_bits)
    stego_pixels = stego_image.astype(np.uint16)
    payload_mask = (1 << payload_bits) - 1
    extracted = (stego_pixels & payload_mask) << (8 - payload_bits)
    return extracted.astype(np.uint8)


def hide_image(
    cover_image_path: str | Path,
    secret_image_path: str | Path,
    output_image_path: str | Path,
    size: int | None = None,
    payload_bits: int = 4,
    method: str = "lsb",
    payload_rate: float = 1.0,
    seed: int | None = None,
) -> None:
    cover_image = read_image(cover_image_path, size=size)
    secret_image = read_image(secret_image_path, size=size)
    stego_image = hide_array(
        cover_image,
        secret_image,
        payload_bits=payload_bits,
        method=method,
        payload_rate=payload_rate,
        seed=seed,
    )
    write_image(output_image_path, stego_image)


def extract_hidden_image(
    stego_image_path: str | Path,
    output_image_path: str | Path,
    size: int | None = None,
    payload_bits: int = 4,
) -> None:
    stego_image = read_image(stego_image_path, size=size)
    extracted_image = extract_array(stego_image, payload_bits=payload_bits)
    write_image(output_image_path, extracted_image)


def make_stego_pairs(
    clean_paths: Iterable[Path],
    clean_out_dir: str | Path | None,
    stego_out_dir: str | Path,
    size: int = 128,
    prefix: str = "",
    payload_bits: int = 4,
    save_clean: bool = True,
    method: str = "lsb",
    payload_rate: float = 1.0,
    seed: int = 42,
) -> int:
    """Copy clean images and create stego images using another image as secret."""
    validate_payload_bits(payload_bits)
    validate_method(method)
    validate_payload_rate(payload_rate)
    clean_paths = list(clean_paths)
    if len(clean_paths) < 2:
        raise ValueError("At least two images are needed to create stego pairs.")

    clean_out_dir = Path(clean_out_dir) if clean_out_dir is not None else None
    stego_out_dir = Path(stego_out_dir)
    if save_clean and clean_out_dir is not None:
        clean_out_dir.mkdir(parents=True, exist_ok=True)
    stego_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, cover_path in enumerate(tqdm(clean_paths, desc=f"{prefix} stego")):
        secret_path = clean_paths[(idx + 1) % len(clean_paths)]
        filename = f"{prefix}_{idx:05d}.png" if prefix else f"{idx:05d}.png"

        cover_image = read_image(cover_path, size=size)
        secret_image = read_image(secret_path, size=size)
        stego_image = hide_array(
            cover_image,
            secret_image,
            payload_bits=payload_bits,
            method=method,
            payload_rate=payload_rate,
            seed=seed + idx,
        )

        if save_clean and clean_out_dir is not None:
            write_image(clean_out_dir / filename, cover_image)
        write_image(stego_out_dir / filename, stego_image)

    return len(clean_paths)
