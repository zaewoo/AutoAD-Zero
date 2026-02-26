import os
import cv2
import sys
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from qwen_vl_utils import smart_resize

# Base setup from qwen2-vl
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def fetch_video_from_tensor(video, image_factor: int = IMAGE_FACTOR):
    nframes, _, height, width = video.shape
    ele = {}
    ele["max_pixels"] = 360 * 420
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video

def process_video(video_path, num_frames=32, sample_scheme='uniform', start = None, end = None):
    decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    start_frame, end_frame = local_fps * start, local_fps * end
    end_frame = min(end_frame, len(decord_vr) - 1)
    frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
    return images, decord_vr

def convert_bounding_box_to_rectangle(bbox, canvas_width=1280, canvas_height=720, line_width=10):
    """
    Input: bbox, canvas (image) size
    Output: binary bbox mask
    """
    x0, y0, x1, y1 = bbox

    # extend the bbox by 20%
    delta_x = (x1 - x0) * 0.1
    delta_y = (y1 - y0) * 0.1

    x0 = int(max(x0 - delta_x, 0))
    y0 = int(max(y0 - delta_y, 0))
    x1 = int(min(x1 + delta_x, canvas_width))
    y1 = int(min(y1 + delta_y, canvas_height))
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas[y0:y0+line_width, x0:x1] = 1. 
    canvas[y1-line_width:y1, x0:x1] = 1. 
    canvas[y0:y1, x0:x0+line_width] = 1.  
    canvas[y0:y1, x1-line_width:x1] = 1. 
    return canvas

def convert_bounding_box_to_ellipse(bbox, canvas_width=1280, canvas_height=720, line_width=10):
    """
    Input: bbox, canvas (image) size
    Output: binary ellipse mask
    """
    x0, y0, x1, y1 = bbox
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    # extend the ellipse by 25%
    axis_x = int(abs(x1 - x0) // 2 * 1.25)
    axis_y = int(abs(y1 - y0) // 2 * 1.25)
    
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas = cv2.ellipse(canvas, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1, line_width)
    return canvas

def process_text_prompt(question, processor):
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ""},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return text, question