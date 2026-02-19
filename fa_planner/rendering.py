import math
from PIL import Image, ImageDraw
import numpy as np


def world_to_pixel(pos, size):
    x, y = pos
    px = int((x + 1.0) * 0.5 * (size - 1))
    py = int((1.0 - (y + 1.0) * 0.5) * (size - 1))
    return px, py


def draw_circle(draw, center, radius, color, fill=True):
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    if fill:
        draw.ellipse(bbox, outline=color, fill=color)
    else:
        draw.ellipse(bbox, outline=color, width=1)


def draw_cross(draw, center, radius, color):
    x, y = center
    draw.line([x - radius, y, x + radius, y], fill=color, width=1)
    draw.line([x, y - radius, x, y + radius], fill=color, width=1)


def draw_arrow(draw, start, vec, color, scale=10.0):
    sx, sy = start
    vx, vy = vec
    ex = sx + vx * scale
    ey = sy - vy * scale
    draw.line([sx, sy, ex, ey], fill=color, width=1)
    angle = math.atan2(sy - ey, ex - sx)
    head_len = 3
    left = (ex - head_len * math.cos(angle - math.pi / 6),
            ey + head_len * math.sin(angle - math.pi / 6))
    right = (ex - head_len * math.cos(angle + math.pi / 6),
             ey + head_len * math.sin(angle + math.pi / 6))
    draw.line([ex, ey, left[0], left[1]], fill=color, width=1)
    draw.line([ex, ey, right[0], right[1]], fill=color, width=1)


def draw_v_marker(draw, center, direction, length=8.0, width=6.0, color=(220, 120, 90), line_width=1):
    dx, dy = direction
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy = 0.0, -1.0
        norm = 1.0
    dx /= norm
    dy /= norm
    half = length * 0.5
    tip = (center[0] + dx * half, center[1] + dy * half)
    base = (center[0] - dx * half, center[1] - dy * half)
    px, py = -dy, dx
    half_w = width * 0.5
    left = (base[0] + px * half_w, base[1] + py * half_w)
    right = (base[0] - px * half_w, base[1] - py * half_w)
    draw.line([tip, left], fill=color, width=line_width)
    draw.line([tip, right], fill=color, width=line_width)


def render_world_frame(
    size,
    agent_pos,
    goal_pos,
    velocity,
    show_goal=True,
    region_colors=None,
    agent_color=(220, 120, 90),
    goal_color=(90, 170, 120),
    boundary_color=(170, 170, 170),
):
    if region_colors:
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)
        bands = len(region_colors)
        for idx, color in enumerate(reversed(region_colors)):
            y0 = int(idx * size / bands)
            y1 = int((idx + 1) * size / bands) - 1
            draw.rectangle([0, y0, size - 1, y1], fill=tuple(color))
    else:
        img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # boundary
    draw.rectangle([0, 0, size - 1, size - 1], outline=boundary_color, width=1)
    # goal
    if show_goal:
        goal_px = world_to_pixel(goal_pos, size)
        draw_cross(draw, goal_px, radius=3, color=goal_color)
    # agent
    agent_px = world_to_pixel(agent_pos, size)
    vx, vy = velocity
    direction = (vx, -vy)
    draw_v_marker(draw, agent_px, direction, length=8.0, width=6.0, color=agent_color, line_width=1)
    return np.array(img, dtype=np.uint8)


def crop_and_resize(img, center_px, crop_size, out_size):
    h, w, _ = img.shape
    cx, cy = center_px
    half = crop_size // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    crop = img[y0:y1, x0:x1]
    pil = Image.fromarray(crop)
    pil = pil.resize((out_size, out_size), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)
