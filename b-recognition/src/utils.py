import os
import copy
import torch
import tarfile
import numpy as np
from PIL import Image
from io import BytesIO

def process_video(decord_vr, num_frames=32, start = None, end = None):
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    start_frame, end_frame = local_fps * start, local_fps * end
    end_frame = min(end_frame, len(decord_vr) - 1)
    frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()    
    images = [f.numpy() if isinstance(f, torch.Tensor) else f for f in video_data]
    return images

def group_and_filter_id(pred_id_all_frames, pred_cos_all_frames, score_threshold):
    index_filtered_dict = {}
    for frame_idx, pred_cos_per_frame in enumerate(pred_cos_all_frames):
        pred_id_per_frame = pred_id_all_frames[frame_idx]
        recorded_id_dict = {}
        for face_idx, pred_cos in enumerate(pred_cos_per_frame):
            if pred_cos < score_threshold: # ignore faces with low cosine similarity (unconfident recognition)
                continue
            pred_id = pred_id_per_frame[face_idx]
            if pred_id not in recorded_id_dict.keys():
                recorded_id_dict[pred_id] = [pred_cos, (frame_idx, face_idx)]
            else:
                if pred_cos > recorded_id_dict[pred_id][0]: # match the face with the character based on highest cosine similarity
                    recorded_id_dict[pred_id] = [pred_cos, (frame_idx, face_idx)]
        # collect up information in this frame and add it to the final result
        for result_id, result_value in recorded_id_dict.items():
            if result_id not in index_filtered_dict.keys():
                index_filtered_dict[result_id] = [result_value[1]]
            else:
                index_filtered_dict[result_id].append(result_value[1])
    return index_filtered_dict
