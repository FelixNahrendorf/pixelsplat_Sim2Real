from PIL import Image, ImageDraw, ImageFont
import os
import re
import numpy as np
import subprocess
from collections import defaultdict

root = "/app/felix/data/pixelsplat_Sim2Real/outputs/gif/Experiment/seed"


os.makedirs(root, exist_ok=True)

CAM_NAMES = {
    0: "FRONT",
    1: "FRONT_RIGHT",
    2: "FRONT_LEFT",
    3: "BACK",
    4: "BACK_LEFT",
    5: "BACK_RIGHT",
}

STEM_LABELS = {
    ("color", "000072"): "POV rendered",
    ("color", "000098"): "BEV rendered",
    ("depth", "000072"): "POV depth rendered",
    ("depth", "000098"): "BEV depth rendered",
}

CAM_LABELS = {
    "FRONT":       "Front",
    "FRONT_RIGHT": "Front Right",
    "FRONT_LEFT":  "Front Left",
    "BACK":        "Back",
    "BACK_LEFT":   "Back Left",
    "BACK_RIGHT":  "Back Right",
}

ROW1_CAMS  = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_RIGHT", "BACK", "BACK_LEFT"]
ROW2_STEMS = [
    ("color", "000072"),   # POV rendered
    ("depth", "000072"),   # POV depth rendered
    ("color", "000098"),   # BEV rendered
]


# ── helpers ───────────────────────────────────────────────────────────────────

def add_label(img: Image.Image, text: str) -> Image.Image:
    """Draw a black-background label flush to the top center of the image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad    = 2
    x      = (img.width - text_w) // 2
    y      = 0  # flush to top

    draw.rectangle([x - pad, 0, x + text_w + pad, text_h + pad * 2], fill="black")
    draw.text((x, y), text, font=font, fill="white")
    return img


def add_label_center(img: Image.Image, text: str) -> Image.Image:
    """Draw a black-background label at the horizontal center of the image,
    flush to the top — used for the grid composite."""
    return add_label(img, text)


def save_gif(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=160, loop=0)
    print(f"Saved {path}")


def gif_to_mp4(gif_path: str, mp4_path: str, fps: int = 5):
    try:
        import imageio
        gif = Image.open(gif_path)
        frames = []
        for i in range(gif.n_frames):
            gif.seek(i)
            frames.append(np.array(gif.convert("RGB")))
        writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Saved {mp4_path}")
    except Exception as e:
        print(f"  imageio error: {e}, falling back to ffmpeg mpeg4 …")
        cmd = [
            "ffmpeg", "-y",
            "-i", gif_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-r", str(fps),
            mp4_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr}")
        else:
            print(f"Saved {mp4_path}")


def build_grid_frame(row1_imgs, row2_imgs):
    r1_w = row1_imgs[0].width
    r1_h = row1_imgs[0].height
    row1_total_w = r1_w * len(row1_imgs)

    r2_w = row2_imgs[0].width * 2
    r2_h = row2_imgs[0].height * 2
    #r2_w = int(row2_imgs[0].width * 0.5) #for BEV
    #r2_h = int(row2_imgs[0].height * 0.5) #for BEV
    row2_total_w = r2_w * len(row2_imgs)

    grid_w = max(row1_total_w, row2_total_w)
    grid_h = r1_h + r2_h
    grid   = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    row1_offset_x = (grid_w - row1_total_w) // 2
    row2_offset_x = (grid_w - row2_total_w) // 2

    for i, img in enumerate(row1_imgs):
        grid.paste(img, (row1_offset_x + i * r1_w, 0))

    for i, img in enumerate(row2_imgs):
        grid.paste(img.resize((r2_w, r2_h), Image.LANCZOS), (row2_offset_x + i * r2_w, r1_h))

    return grid


def label_grid_frame(grid: Image.Image, row1_labels, row2_labels,
                     r1_w, r1_h, r2_w, r2_h,
                     row1_offset_x, row2_offset_x) -> Image.Image:
    """Apply labels onto an already-built grid frame."""
    grid = grid.copy()
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    def draw_label(text, cell_x, cell_y, cell_w):
        bbox   = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad    = 2
        x      = cell_x + (cell_w - text_w) // 2
        y      = cell_y  # flush to top of cell
        draw.rectangle([x - pad, y, x + text_w + pad, y + text_h + pad * 2], fill="black")
        draw.text((x, y), text, font=font, fill="white")

    for i, label in enumerate(row1_labels):
        draw_label(label, row1_offset_x + i * r1_w, 0, r1_w)

    for i, label in enumerate(row2_labels):
        draw_label(label, row2_offset_x + i * r2_w, r1_h, r2_w)

    return grid


# ── discover scenes ───────────────────────────────────────────────────────────

scene_dirs = [
    d for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d)) and re.match(r'scene-\d+_\d+', d)
]

unique_scenes = defaultdict(list)
for d in scene_dirs:
    match = re.match(r'(scene-\d+)_(\d+)', d)
    if match:
        unique_scenes[match.group(1)].append((int(match.group(2)), d))

for scene_name in unique_scenes:
    unique_scenes[scene_name].sort(key=lambda x: x[0])

gif_out_dir         = os.path.join(root, "gif")
gif_out_dir_labeled = os.path.join(root, "gif", "with_labels")
os.makedirs(gif_out_dir, exist_ok=True)
os.makedirs(gif_out_dir_labeled, exist_ok=True)

all_scenes_sorted = sorted(unique_scenes.keys())
CAM_IDX = {v: k for k, v in CAM_NAMES.items()}


# ── per-scene GIFs ────────────────────────────────────────────────────────────

for scene_name, indexed_dirs in unique_scenes.items():

    # color & depth
    for subdir in ["color", "depth"]:
        target_stems = set()
        for idx, dir_name in indexed_dirs:
            frame_dir = os.path.join(root, dir_name, subdir)
            if not os.path.isdir(frame_dir):
                continue
            for f in os.listdir(frame_dir):
                if f.endswith(".png"):
                    target_stems.add(os.path.splitext(f)[0])

        for stem in sorted(target_stems):
            label          = STEM_LABELS.get((subdir, stem), f"{subdir} {stem}")
            frames_plain   = []
            frames_labeled = []

            for idx, dir_name in indexed_dirs:
                img_path = os.path.join(root, dir_name, subdir, f"{stem}.png")
                if not os.path.exists(img_path):
                    continue
                img = Image.open(img_path).convert("RGB")
                frames_plain.append(img.copy())
                frames_labeled.append(add_label(img, label))

            if not frames_plain:
                continue

            gif_name = f"{scene_name}_{subdir}_{stem}.gif"
            save_gif(frames_plain,   os.path.join(gif_out_dir,         gif_name))
            save_gif(frames_labeled, os.path.join(gif_out_dir_labeled, gif_name))

    # reference (6 cameras)
    ref_frames_plain   = defaultdict(list)
    ref_frames_labeled = defaultdict(list)

    for idx, dir_name in indexed_dirs:
        frame_dir = os.path.join(root, dir_name, "reference")
        if not os.path.isdir(frame_dir):
            continue
        for png in sorted(f for f in os.listdir(frame_dir) if f.endswith(".png")):
            cam_idx  = int(os.path.splitext(png)[0])
            cam_name = CAM_NAMES.get(cam_idx, f"cam{cam_idx}")
            label    = CAM_LABELS.get(cam_name, cam_name)
            img      = Image.open(os.path.join(frame_dir, png)).convert("RGB")
            ref_frames_plain[cam_idx].append(img.copy())
            ref_frames_labeled[cam_idx].append(add_label(img, label))

    for cam_idx in ref_frames_plain:
        cam_name = CAM_NAMES.get(cam_idx, f"cam{cam_idx}")
        gif_name = f"{scene_name}_reference_{cam_name}.gif"
        save_gif(ref_frames_plain[cam_idx],   os.path.join(gif_out_dir,         gif_name))
        save_gif(ref_frames_labeled[cam_idx], os.path.join(gif_out_dir_labeled, gif_name))


# ── full concatenated GIFs ────────────────────────────────────────────────────

print("\nBuilding full concatenated GIFs …")

for subdir in ["color", "depth"]:
    all_stems = set()
    for scene_name in all_scenes_sorted:
        for idx, dir_name in unique_scenes[scene_name]:
            frame_dir = os.path.join(root, dir_name, subdir)
            if not os.path.isdir(frame_dir):
                continue
            for f in os.listdir(frame_dir):
                if f.endswith(".png"):
                    all_stems.add(os.path.splitext(f)[0])

    for stem in sorted(all_stems):
        for out_dir in [gif_out_dir, gif_out_dir_labeled]:
            all_frames = []
            for scene_name in all_scenes_sorted:
                gif_path = os.path.join(out_dir, f"{scene_name}_{subdir}_{stem}.gif")
                if not os.path.exists(gif_path):
                    continue
                g = Image.open(gif_path)
                for i in range(g.n_frames):
                    g.seek(i)
                    all_frames.append(g.convert("RGB").copy())
            if not all_frames:
                continue
            save_gif(all_frames, os.path.join(out_dir, f"ALL_{subdir}_{stem}.gif"))

for cam_idx, cam_name in CAM_NAMES.items():
    for out_dir in [gif_out_dir, gif_out_dir_labeled]:
        all_frames = []
        for scene_name in all_scenes_sorted:
            gif_path = os.path.join(out_dir, f"{scene_name}_reference_{cam_name}.gif")
            if not os.path.exists(gif_path):
                continue
            g = Image.open(gif_path)
            for i in range(g.n_frames):
                g.seek(i)
                all_frames.append(g.convert("RGB").copy())
        if not all_frames:
            continue
        save_gif(all_frames, os.path.join(out_dir, f"ALL_reference_{cam_name}.gif"))


# ── grid GIFs + MP4 ───────────────────────────────────────────────────────────

print("\nBuilding grid GIFs …")

# precompute label lists for the grid
row1_label_list = [CAM_LABELS[c] for c in ROW1_CAMS]
row2_label_list = [STEM_LABELS.get((s, st), f"{s} {st}") for s, st in ROW2_STEMS]

for use_labels in [False, True]:
    out_dir = gif_out_dir_labeled if use_labels else gif_out_dir
    suffix  = "_labeled" if use_labels else ""

    # per-scene grid — always built from plain images
    for scene_name, indexed_dirs in unique_scenes.items():
        grid_frames = []

        # need sizes for label_grid_frame; read from first available frame
        r1_w = r1_h = r2_w_base = r2_h_base = None

        for idx, dir_name in indexed_dirs:
            missing = False

            row1_imgs = []
            for cam_name in ROW1_CAMS:
                img_path = os.path.join(root, dir_name, "reference", f"{CAM_IDX[cam_name]:06d}.png")
                if not os.path.exists(img_path):
                    print(f"  Missing {img_path}, skipping grid frame.")
                    missing = True
                    break
                row1_imgs.append(Image.open(img_path).convert("RGB"))
            if missing:
                continue

            row2_imgs = []
            for subdir, stem in ROW2_STEMS:
                img_path = os.path.join(root, dir_name, subdir, f"{stem}.png")
                if not os.path.exists(img_path):
                    print(f"  Missing {img_path}, skipping grid frame.")
                    missing = True
                    break
                row2_imgs.append(Image.open(img_path).convert("RGB"))
            if missing:
                continue

            # capture sizes once
            if r1_w is None:
                r1_w, r1_h       = row1_imgs[0].width, row1_imgs[0].height
                r2_w_base        = row2_imgs[0].width * 2
                r2_h_base        = row2_imgs[0].height * 2

            grid = build_grid_frame(row1_imgs, row2_imgs)

            if use_labels:
                row1_total_w = r1_w * len(ROW1_CAMS)
                row2_total_w = r2_w_base * len(ROW2_STEMS)
                grid_w       = max(row1_total_w, row2_total_w)
                row1_off     = (grid_w - row1_total_w) // 2
                row2_off     = (grid_w - row2_total_w) // 2
                grid = label_grid_frame(
                    grid,
                    row1_label_list, row2_label_list,
                    r1_w, r1_h, r2_w_base, r2_h_base,
                    row1_off, row2_off,
                )

            grid_frames.append(grid)

        if not grid_frames:
            continue
        save_gif(grid_frames, os.path.join(out_dir, f"{scene_name}_grid{suffix}.gif"))

    # ALL grid GIF + MP4
    all_grid_frames = []
    for scene_name in all_scenes_sorted:
        gif_path = os.path.join(out_dir, f"{scene_name}_grid{suffix}.gif")
        if not os.path.exists(gif_path):
            print(f"  Missing {gif_path}, skipping.")
            continue
        g = Image.open(gif_path)
        for i in range(g.n_frames):
            g.seek(i)
            all_grid_frames.append(g.convert("RGB").copy())

    if all_grid_frames:
        all_grid_gif = os.path.join(out_dir, f"ALL_grid{suffix}.gif")
        save_gif(all_grid_frames, all_grid_gif)
        gif_to_mp4(all_grid_gif, all_grid_gif.replace(".gif", ".mp4"), fps=5)

print("\nDone.")