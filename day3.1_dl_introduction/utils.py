import math, os, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import openslide
from torchvision import transforms
from model import SimpleCNN
import numpy as np
import torchstain
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
import math, numpy as np, torch, random


def plot_normalization_example(base_ds, img_size, idx):
    path, label = base_ds.samples[idx]
    pil = base_ds.loader(path)

    to_tensor = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor()])  # in [0,1]
    x = to_tensor(pil)                # shape [3, H, W]
    x_flat = x.view(3, -1)            # [3, H*W]

    mean = x_flat.mean(dim=1, keepdim=True)  # [3,1]
    std  = x_flat.std(dim=1, unbiased=False, keepdim=True) + 1e-8

    x_norm = (x - mean.view(3,1,1)) / std.view(3,1,1)
    x_norm_flat = x_norm.view(3, -1)

    fig = plt.figure(figsize=(12,3))

    ax1 = plt.subplot(1,3,1)
    ax1.imshow(np.moveaxis(x.numpy(), 0, 2))
    ax1.set_title("Original image")
    ax1.axis("off")

    ax2 = plt.subplot(1,3,2)
    colors = ["red","green","blue"]
    names  = ["R","G","B"]
    for c,(name,col) in enumerate(zip(names, colors)):
        ax2.hist(x_flat[c].numpy(), bins=40, range=(0,1),
                 alpha=0.55, density=True, color=col, label=name)
        ax2.axvline(x_flat[c].mean().item(), color=col, linestyle="--", linewidth=1)
    ax2.set_title("BEFORE (values ~0..1)")
    ax2.set_xlabel("value"); ax2.set_ylabel("density")
    ax2.legend()

    ax3 = plt.subplot(1,3,3)
    for c,(name,col) in enumerate(zip(names, colors)):
        ax3.hist(x_norm_flat[c].numpy(), bins=40, range=(-3,3),
                 alpha=0.55, density=True, color=col, label=name)
    # lines at 0 mean and ±1 std
    ax3.axvline(0, color="black", linestyle="--", linewidth=1)
    ax3.axvline(1, color="black", linestyle=":", linewidth=1)
    ax3.axvline(-1, color="black", linestyle=":", linewidth=1)
    ax3.set_title("AFTER (≈ mean 0, std 1)")
    ax3.set_xlabel("z-score"); ax3.set_ylabel("density")
    ax3.legend()

    plt.tight_layout()
    plt.show()

def show_topk_for_classes(dataset, class_names, all_probs, all_preds, all_true,
                          k=6, mode="target", only=None, resize=224):
    # ensure NumPy arrays (robust if inputs are torch tensors)
    to_np = lambda a: a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    probs = to_np(all_probs)    # (N, C)
    preds = to_np(all_preds)    # (N,)
    true  = to_np(all_true)     # (N,)

    classes = dataset.classes
    paths   = [p for p,_ in dataset.samples]
    N = len(paths)
    assert probs.shape[0] == preds.shape[0] == true.shape[0] == N

    sel_idx = [classes.index(n) for n in class_names]
    fig, axes = plt.subplots(len(sel_idx), k, figsize=(3*k, 3*len(sel_idx)), squeeze=False)

    for r, ci in enumerate(sel_idx):
        scores = probs[:, ci]
        cand = np.where(preds == ci)[0] if mode == "pred" else np.arange(N, dtype=np.int64)
        if only == "tp": cand = cand[true[cand] == ci]
        elif only == "fp": cand = cand[true[cand] != ci]

        if cand.size == 0 or k <= 0:
            for c in range(k): axes[r, c].axis("off")
            axes[r, 0].set_title(f"{classes[ci]} (no samples)", loc="left")
            continue

        m = min(k, cand.size)
        # robust top-k without negative-step slices
        sel   = np.argpartition(-scores[cand], m-1)[:m]   # indices within cand
        order = np.argsort(-scores[cand][sel])            # sort those m by score desc
        top   = cand[sel[order]]

        for c, idx in enumerate(top):
            ax = axes[r, c]
            img = Image.open(paths[idx]).convert("RGB")
            if resize: img = img.resize((resize, resize))
            ax.imshow(img); ax.axis("off")
            ok = (preds[idx] == true[idx])
            ax.set_title(f"{classes[ci]}  p={scores[idx]:.2f}\n"
                         f"{'✓' if ok else '✗'} pred={classes[int(preds[idx])]}, true={classes[int(true[idx])]}",
                         fontsize=9)
        for c in range(top.size, k): axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

BETA_OD   = 0.15                # OD threshold used inside Macenko & mask
MICROBATCH= 32                  # tiles per forward pass
USE_RAW_MAP = True              # set False to skip raw branch & double speed

def ensure_chw(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 2:
        img = img.unsqueeze(0).repeat(3,1,1)
    elif img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        img = img.permute(2,0,1)
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    return img

def T255(pil_img: Image.Image) -> torch.Tensor:
    # Make a writable, C-contiguous uint8 array
    arr = np.array(pil_img, dtype=np.uint8, copy=True)   # <-- forces writable copy
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t

def to_model(x255: torch.Tensor, device='cuda', mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=device).view(3,1,1)
    std_t  = torch.tensor(std,  device=device).view(3,1,1)
    x = ensure_chw(x255).to(device, non_blocking=True).clamp(0,255).float()/255.0
    return ((x - mean_t) / std_t)  # (3,H,W)

def purple_tint(x255: torch.Tensor, *, strength=1.0, noise_std=2.5, rgb=(170, 120, 170)) -> torch.Tensor:
    x255 = ensure_chw(x255)
    _, H, W = x255.shape
    bg = torch.tensor(rgb, dtype=torch.float32, device=x255.device).view(3,1,1).expand(3,H,W)
    y = (1.0 - strength) * x255.float() + strength * bg
    if noise_std and noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    return y.clamp(0,255).to(torch.uint8)

def calc_tile_geometry(w_level, h_level, tile_px, eff_mpp, target_mpp=None):
    """
    Returns: read_size (level px), step (level px), rows, cols
    - If target_mpp is None: read_size==step==tile_px (no resample).
    - Else: read_size = round(tile_px * target_mpp / eff_mpp), then resize to tile_px.
    """
    if target_mpp is None:
        read_size = tile_px
    else:
        read_size = max(1, int(round(tile_px * (target_mpp / eff_mpp))))
    step = read_size
    cols = math.ceil(w_level / step)
    rows = math.ceil(h_level / step)
    return read_size, step, rows, cols

def safe_macenko_or_purple(raw255_gpu: torch.Tensor,
                           *, alpha=1, beta=BETA_OD, min_stain_px=80,
                           bg_strength=1.0, bg_rgb=(170, 120, 170)):
    """
    Fast path: do a cheap blank check on CPU-arr first (already done outside).
    Here we assume tile is NOT blank; try Macenko, else return raw.
    """
    # Convert to CPU NumPy only here (stained tiles)
    np_in = raw255_gpu.permute(1,2,0).contiguous().to("cpu").numpy()  # HxWx3 uint8
    try:
        out = normalizer.normalize(I=np_in, Io=255, alpha=alpha, beta=beta, stains=False)
        img = out[0] if isinstance(out, (tuple,list)) else out
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img)
            if t.ndim == 2:
                t = t.unsqueeze(-1).repeat(1,1,3)
            if t.shape[-1] == 3:
                t = t.permute(2,0,1)
            if t.dtype != torch.uint8:
                t = t.clamp(0,255).to(torch.uint8)
            return t.to(raw255_gpu.device, non_blocking=True)
        elif isinstance(img, torch.Tensor):
            t = ensure_chw(img)
            if t.dtype != torch.uint8:
                t = t.clamp(0,255).to(torch.uint8)
            return t.to(raw255_gpu.device, non_blocking=True)
        else:
            return raw255_gpu
    except Exception:
        return raw255_gpu

def iterate_tiles(slide, level, w_level, h_level, eff_down,
                  tile_px=224, read_size=None, step=None,
                  blank_t=235, min_stain=80, device='cuda'):
    """Yield (r, c, raw255, mac255, has_tissue, (W,H), stained_px)."""
    assert read_size is not None and step is not None, "Pass read_size and step from calc_tile_geometry"
    cols = math.ceil(w_level / step)
    rows = math.ceil(h_level / step)

    for r in range(rows):
        for c in range(cols):
            x_l = c * step
            y_l = r * step
            if x_l >= w_level or y_l >= h_level:
                continue

            # read a square region of size read_size (clip at edges)
            w_tile = min(read_size, w_level - x_l)
            h_tile = min(read_size, h_level - y_l)
            x0 = int(x_l * eff_down)
            y0 = int(y_l * eff_down)

            region = slide.read_region((x0, y0), level, (w_tile, h_tile)).convert("RGB")

            # pad to read_size, then resize to model size (tile_px) if needed
            if (w_tile, h_tile) != (read_size, read_size):
                canvas = Image.new("RGB", (read_size, read_size), (0, 0, 0))
                canvas.paste(region, (0, 0))
                region = canvas
            if read_size != tile_px:
                region = region.resize((tile_px, tile_px), Image.BILINEAR)  # exact 0.5 MPP field-of-view

            arr = np.array(region, dtype=np.uint8, copy=True)   # CPU HxWx3

            # quick blank precheck
            mn = arr.min(axis=2)
            stained_px = int((mn < blank_t).sum())

            raw255 = torch.from_numpy(arr).permute(2,0,1).contiguous().to(device, non_blocking=True)

            if stained_px < min_stain:
                mac255 = purple_tint(raw255, strength=1.0, rgb=(170, 120, 170))
            else:
                mac255 = safe_macenko_or_purple(raw255, beta=BETA_OD, min_stain_px=min_stain)

            yield r, c, raw255, mac255, (stained_px >= min_stain), (arr.shape[1], arr.shape[0]), stained_px

def run_tiled_inference(slide, level, w_level, h_level, eff_down, classes, model,
                        tile_px=224, use_raw_map=True, target_mpp=None):
    model.eval()
    # compute geometry
    eff_mpp = float(slide.properties.get("openslide.mpp-x", 0.25)) * float(eff_down)
    read_size, step, rows, cols = calc_tile_geometry(w_level, h_level, tile_px, eff_mpp, target_mpp)

    C = len(classes)
    probs_map_norm = np.zeros((rows, cols, C), dtype=np.float32)
    probs_map_raw  = np.zeros((rows, cols, C), dtype=np.float32) if use_raw_map else None
    mask_map       = np.zeros((rows, cols), dtype=bool)

    batch_raw, batch_mac, batch_rc = [], [], []

    def flush_batch():
        nonlocal batch_raw, batch_mac, batch_rc
        if not batch_rc:
            return
        with torch.no_grad():
            X_mac = torch.stack([to_model(t) for t in batch_mac], dim=0)
            if USE_RAW_MAP:
                X_raw = torch.stack([to_model(t) for t in batch_raw], dim=0)
                X = torch.cat([X_raw, X_mac], dim=0)
                probs = model(X).softmax(1).detach().cpu().numpy()
                B = len(batch_rc); probs_raw, probs_mac = probs[:B], probs[B:]
            else:
                probs_mac = model(X_mac).softmax(1).detach().cpu().numpy()
                probs_raw = None
        for i, (r, c) in enumerate(batch_rc):
            probs_map_norm[r, c] = probs_mac[i]
            if USE_RAW_MAP:
                probs_map_raw[r, c] = probs_raw[i]
        batch_raw.clear(); batch_mac.clear(); batch_rc.clear()

    total = rows * cols
    for (r,c, raw255, mac255, has_tissue, _, stained_px) in tqdm(
            iterate_tiles(slide, level, w_level, h_level, eff_down,
                          tile_px=tile_px, read_size=read_size, step=step),
            total=total, desc=f"Tessellate@level{level} (read={read_size}, step={step})"):
        mask_map[r, c] = bool(has_tissue)
        batch_raw.append(raw255); batch_mac.append(mac255); batch_rc.append((r, c))
        if len(batch_rc) >= MICROBATCH:
            flush_batch()
    flush_batch()
    return probs_map_norm, probs_map_raw, mask_map

def cap_size(w, h, max_side=512):
    if max(w, h) <= max_side:
        return w, h
    if w >= h:
        return max_side, int(h * (max_side / w))
    else:
        return int(w * (max_side / h)), max_side

def grid_to_vis(grid2d, size_wh=None, level_wh=None, max_side=512):
    g = grid2d
    if g.dtype != np.uint8 and g.max() <= 1.0:
        g = (g.astype(np.float32) * 255).clip(0,255).astype(np.uint8)
    im = Image.fromarray(g)
    if size_wh is not None:
        return im.resize(size_wh, resample=Image.NEAREST)
    if level_wh is not None:
        vis_w, vis_h = cap_size(*level_wh, max_side=max_side)
        return im.resize((vis_w, vis_h), resample=Image.NEAREST)
    return im  # no resize

def small_thumbnail(slide, w_level, h_level):
    vis_w, vis_h = cap_size(w_level, h_level)
    return slide.get_thumbnail((vis_w, vis_h))
