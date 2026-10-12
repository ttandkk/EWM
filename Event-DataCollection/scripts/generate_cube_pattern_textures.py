"""Generate deterministic UV textures for the Piper stack cubes."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

SIZE = 256


def save_red_stripes(path: Path) -> None:
    image = Image.new("RGB", (SIZE, SIZE), (185, 28, 28))
    draw = ImageDraw.Draw(image)
    for x in (28, 104, 180):
        draw.rectangle((x, 0, x + 28, SIZE), fill=(238, 238, 238))
    image.save(path)


def save_blue_cross(path: Path) -> None:
    image = Image.new("RGB", (SIZE, SIZE), (28, 54, 175))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 104, SIZE, 152), fill=(244, 196, 24))
    draw.rectangle((104, 0, 152, SIZE), fill=(244, 196, 24))
    image.save(path)


def save_green_checker(path: Path) -> None:
    image = Image.new("RGB", (SIZE, SIZE), (20, 125, 48))
    draw = ImageDraw.Draw(image)
    cell = 64
    for row in range(4):
        for col in range(4):
            if (row + col) % 2 == 0:
                draw.rectangle((col * cell, row * cell, (col + 1) * cell, (row + 1) * cell), fill=(235, 235, 235))
    image.save(path)


args.output.mkdir(parents=True, exist_ok=True)
save_red_stripes(args.output / "red_stripes.png")
save_blue_cross(args.output / "blue_cross.png")
save_green_checker(args.output / "green_checker.png")
print(f"wrote textures to {args.output}")
