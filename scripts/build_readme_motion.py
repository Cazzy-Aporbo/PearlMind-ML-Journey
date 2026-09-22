"""Render an original, illustrative learning-flow GIF for GitHub's script-free README."""

from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
font_paths = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
font_path = next((p for p in font_paths if Path(p).exists()), None)
font = ImageFont.truetype(font_path, 17) if font_path else ImageFont.load_default()
small = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()
frames = []
colors = [(156, 231, 212), (185, 190, 251), (230, 180, 214), (242, 201, 164)]
for frame in range(48):
    im = Image.new("RGB", (1000, 180), (10, 16, 32))
    d = ImageDraw.Draw(im)
    d.text((30, 16), "A LEARNING LOOP / ILLUSTRATIVE FLOW", font=small, fill=(158, 178, 204))
    xs = [130, 380, 630, 870]
    for j in range(3):
        d.line((xs[j] + 36, 86, xs[j + 1] - 36, 86), fill=(53, 65, 91), width=1)
        t = (frame / 48 + j / 3) % 1
        x = xs[j] + 36 + t * (xs[j + 1] - xs[j] - 72)
        d.ellipse((x - 4, 82, x + 4, 90), fill=colors[j])
    for j, (x, label) in enumerate(zip(xs, ["Observe", "Train", "Hold out", "Question"])):
        c = colors[j]
        d.ellipse((x - 24, 62, x + 24, 110), outline=c, width=2)
        phase = (frame / 48 - j / 4) * math.tau
        px = x + 24 * math.cos(phase)
        py = 86 + 24 * math.sin(phase)
        d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=c)
        d.text((x - 5, 77), str(j + 1), font=small, fill=c)
        d.text((x - d.textlength(label, font=font) / 2, 126), label, font=font, fill=c)
    frames.append(im)
frames[0].save(
    root / "web/assets/learning-loop.gif",
    save_all=True,
    append_images=frames[1:],
    duration=85,
    loop=0,
    optimize=True,
)
print("Rendered learning-loop.gif")
