import os
import sys
import ast
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu
from pandas.compat import pickle_compat
from utils import fetch_video_from_tensor, convert_bounding_box_to_rectangle, convert_bounding_box_to_ellipse, process_text_prompt, process_video

class MADDataLoader():
    def __init__(
        self,
        model, 
        processor,
        general_prompt,
        label_type, 
        label_width, 
        label_alpha,
        anno_path,
        video_dir,
        charbank_path,
        num_frames,
        prompt_idx,
        **kwargs
    ):
        
        self.model = model
        self.processor = processor
        self.general_prompt=general_prompt
        self.num_frames = num_frames
        self.transform = transforms.Resize((336,))
        self.prompt_idx = prompt_idx

        # Label information, including colour coding, type, visual prompts, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha

        # Load source file
        self.anno_df = pd.read_csv(anno_path)

        # Prepare character bank as dictionaries
        self.charbank_dict = {}
        with open(charbank_path, "rb") as f:
            charbank_pred = pickle_compat.load(f)
            charbank_pred = {str(key):value for key,value in charbank_pred.items()}
            for movie_name in charbank_pred.keys():
                charbank_pred_per_frame = charbank_pred[movie_name]
                name_ids_per_frame = charbank_pred_per_frame["charbank_nmids"].iloc[0]
                roles_per_frame = charbank_pred_per_frame["charbank_roles"].iloc[0]
                self.charbank_dict[movie_name] = {k: v for k, v in zip(name_ids_per_frame, roles_per_frame)}

        self.all_clips = []
        for _, anno_row in self.anno_df.iterrows():
            sentence = anno_row["sentence"]
            movie_name = anno_row["movie"]

            # The AD interval
            start = 0 
            end = anno_row["end_seconds"] - anno_row["start_seconds"]

            # Video path
            imdbid = anno_row["id"]
            lsmdc_filename = anno_row["path"]
            video_path = os.path.join(video_dir,movie_name,lsmdc_filename+'.avi')

            if os.path.exists(video_path):
                self.all_clips.append((imdbid, movie_name, video_path, start, end, anno_row["start"], anno_row["end"], sentence, anno_row["bboxes"], anno_row["characters"]))
        print(f"In total {len(self.all_clips)} MAD-Eval clips")

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        imdbid, movie_name, video_path, start, end, start_, end_, gt_text, bboxes, name_ids = self.all_clips[index]

        frames, decord_vr = process_video(
            video_path,
            num_frames=self.num_frames,
            sample_scheme='uniform',
            start=start,
            end=end,
        )

        bboxes = ast.literal_eval(bboxes)
        name_ids = ast.literal_eval(name_ids)

        # Stored annotations are generated on 32 sampled frames.
        # If Stage 1 uses a different frame count (e.g., 8/16), keep only
        # the annotation indices that correspond to the sampled input frames.
        anno_frame_indices = np.linspace(0, len(bboxes), self.num_frames, endpoint=False, dtype=int)
        sampled_anno_idx_to_frame_idx = {
            anno_idx: sampled_idx for sampled_idx, anno_idx in enumerate(anno_frame_indices)
        }

        bboxes_filtered = []
        all_name_ids = {}
        for frame_idx in range(len(frames)):
            bboxes_filtered_per_frame = {}
            for name_idx, (name_id, bbox_idx_list) in enumerate(name_ids.items()):
                for bbox_idx in bbox_idx_list:
                    sampled_frame_idx = sampled_anno_idx_to_frame_idx.get(bbox_idx[0])
                    if sampled_frame_idx != frame_idx or name_id in bboxes_filtered_per_frame.keys():
                        continue
                    if bbox_idx[0] >= len(bboxes) or bbox_idx[1] >= len(bboxes[bbox_idx[0]]):
                        continue
                    original_bbox = bboxes[bbox_idx[0]][bbox_idx[1]]
                    rescaled_bbox = [int(coord * 480 / 720) for coord in original_bbox]
                    bboxes_filtered_per_frame[name_id] = rescaled_bbox
                    if name_id not in all_name_ids.keys():
                        all_name_ids[name_id] = len(all_name_ids)
            bboxes_filtered.append(bboxes_filtered_per_frame)

        processed_frames = []
        for frame_idx, frame in enumerate(frames):
            if len(bboxes_filtered[frame_idx]) == 0:
                processed_frames.append(frame)
            else:
                label_masks = None
                total_masks = None
                for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                    if self.label_type=="boxes":
                        label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                    elif self.label_type=="circles":
                        label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                    else:
                        print("Check the label type")
                        sys.exit()
                    if label_masks is None:
                        label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                        total_masks = label_mask
                    else:
                        label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                        total_masks = np.clip(label_mask + total_masks, 0., 1.)
                processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))            
                processed_frames.append(processed_frame)

        process_frames = np.stack(processed_frames, 0)
        video_tensor = torch.from_numpy(process_frames).permute(0, 3, 1, 2)
        video = fetch_video_from_tensor(video_tensor)

        '''
        Formulate character information and text prompts
        '''
        charbank_dict = self.charbank_dict[movie_name]
        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending 
        if char_text == ". Possible characters (labeled by {label_type}): " or self.label_type == "none": 
            char_text = ""
        else:
            char_text=char_text.format(label_type=self.label_type)

        processed_text_prompt, text_prompt = process_text_prompt(self.general_prompt.apply(char_text=char_text), self.processor)

        return_dict =  {
            'video': video,
            'imdbid': imdbid,
            'processed_prompt': processed_text_prompt,
            'prompt': text_prompt,
            'gt_text': gt_text,
            'start': start,  
            'end': end, 
            'start_': start_, 
            'end_': end_,
        }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['processed_prompt'] = [sample['processed_prompt'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        return out_batch
