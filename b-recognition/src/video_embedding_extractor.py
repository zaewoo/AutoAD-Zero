import argparse
import csv
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


SUPPORTED_EXTS = {".avi", ".mp4", ".mkv", ".mov"}


def to_feature_tensor(model_output) -> torch.Tensor:
    """
    Convert heterogeneous HF model outputs to a single 2D feature tensor [B, D].
    This avoids version/model-specific return type issues.
    """
    if isinstance(model_output, torch.Tensor):
        features = model_output
    elif hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        # Common container returned by vision encoders
        features = model_output.pooler_output
    elif hasattr(model_output, "last_hidden_state") and model_output.last_hidden_state is not None:
        # Fallback: CLS token representation
        features = model_output.last_hidden_state[:, 0, :]
    elif isinstance(model_output, (tuple, list)) and len(model_output) > 0 and isinstance(model_output[0], torch.Tensor):
        features = model_output[0]
    else:
        raise TypeError(f"Unsupported model output type for feature extraction: {type(model_output)}")

    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2:
        raise ValueError(f"Expected [B, D] features, got shape={tuple(features.shape)}")
    return features


def discover_videos(video_root: Path) -> List[Path]:
    videos = [
        p for p in video_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(videos)


def sample_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames: List[Image.Image] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        raise RuntimeError(f"All sampled frame reads failed: {video_path}")

    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    return frames[:num_frames]


class VideoEmbeddingExtractor:
    """Video-level embedding extractor (e.g., X-CLIP)."""

    @staticmethod
    def _extract_pixel_values(candidate_inputs) -> torch.Tensor | None:
        for key in ("pixel_values", "pixel_values_videos", "video_pixel_values"):
            value = candidate_inputs.get(key)
            if isinstance(value, torch.Tensor):
                return value

        tensor_values = [v for v in candidate_inputs.values() if isinstance(v, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]
        return None

    def __init__(self, model_name: str, device: str):
        self.model_type = "video"
        self.pooling = "none"
        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            self.image_processor = None
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_video(self, frames: List[Image.Image]) -> np.ndarray:
        inputs = None
        pixel_values = None

        # 1) Try video-aware processor signatures first.
        for processor_kwargs in ({"videos": [frames]}, {"videos": frames}):
            try:
                candidate_inputs = self.processor(**processor_kwargs, return_tensors="pt")
            except Exception:
                continue
            candidate_pixels = self._extract_pixel_values(candidate_inputs)
            if isinstance(candidate_pixels, torch.Tensor):
                inputs = candidate_inputs
                pixel_values = candidate_pixels
                break

        # 2) Try image-preprocessing pathways and convert to [1, T, C, H, W].
        if pixel_values is None:
            image_preprocessors = [
                self.image_processor,
                getattr(self.processor, "image_processor", None),
                getattr(self.processor, "feature_extractor", None),
                self.processor,
            ]
            for image_preprocessor in image_preprocessors:
                if image_preprocessor is None:
                    continue
                try:
                    image_inputs = image_preprocessor(images=frames, return_tensors="pt")
                except Exception:
                    continue
                image_pixels = self._extract_pixel_values(image_inputs)
                if not isinstance(image_pixels, torch.Tensor):
                    continue
                if image_pixels.ndim == 4:
                    pixel_values = image_pixels.unsqueeze(0)
                elif image_pixels.ndim == 5:
                    pixel_values = image_pixels
                else:
                    continue
                inputs = {"pixel_values": pixel_values}
                break

        # 3) Last-resort: per-frame preprocessing + stack.
        if pixel_values is None:
            image_preprocessor = (
                self.image_processor
                or getattr(self.processor, "image_processor", None)
                or getattr(self.processor, "feature_extractor", None)
            )
            if image_preprocessor is not None:
                per_frame_tensors = []
                for frame in frames:
                    try:
                        frame_inputs = image_preprocessor(images=frame, return_tensors="pt")
                    except Exception:
                        continue
                    frame_pixels = frame_inputs.get("pixel_values")
                    if isinstance(frame_pixels, torch.Tensor) and frame_pixels.ndim == 4 and frame_pixels.shape[0] == 1:
                        per_frame_tensors.append(frame_pixels[0])
                if per_frame_tensors:
                    pixel_values = torch.stack(per_frame_tensors, dim=0).unsqueeze(0)
                    inputs = {"pixel_values": pixel_values}

        if inputs is None or pixel_values is None:
            raise RuntimeError(
                "Failed to construct video pixel tensor from processor output. "
                "Check processor/model compatibility for the selected video model."
            )

        # Force canonical key expected by video model APIs.
        inputs = {k: v for k, v in inputs.items() if v is not None}
        inputs["pixel_values"] = pixel_values
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if hasattr(self.model, "get_video_features"):
            features = self.model.get_video_features(pixel_values=inputs["pixel_values"])
        else:
            features = self.model(**inputs)

        features = to_feature_tensor(features)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features[0].cpu().numpy().astype(np.float32)



class FrameAverageEmbeddingExtractor:
    """Frame-level embedding extractor with temporal average pooling."""

    def __init__(self, model_name: str, device: str):
        self.model_type = "frame"
        self.pooling = "frame_mean"
        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_video(self, frames: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        if hasattr(self.model, "get_image_features"):
            frame_features = self.model.get_image_features(pixel_values=pixel_values)
        else:
            frame_features = self.model(pixel_values=pixel_values)

        frame_features = to_feature_tensor(frame_features)
        frame_features = torch.nn.functional.normalize(frame_features, p=2, dim=-1)

        video_feature = frame_features.mean(dim=0, keepdim=True)
        video_feature = torch.nn.functional.normalize(video_feature, p=2, dim=-1)
        return video_feature[0].cpu().numpy().astype(np.float32)


def save_embedding(
    output_root: Path,
    video_root: Path,
    video_path: Path,
    embedding: np.ndarray,
    embedding_type: str,
) -> Path:
    rel_path = video_path.relative_to(video_root)
    target_path = (output_root / embedding_type / rel_path).with_suffix(".npy")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, embedding)
    return target_path


def run(args: argparse.Namespace) -> None:
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    videos = discover_videos(video_root)

    extractors = {}
    if args.embedding_mode in {"video", "both"}:
        extractors["video"] = VideoEmbeddingExtractor(model_name=args.video_model_name, device=device)
    if args.embedding_mode in {"frame", "both"}:
        extractors["frame"] = FrameAverageEmbeddingExtractor(model_name=args.frame_model_name, device=device)

    manifest_path = output_root / "embedding_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_path",
            "embedding_type",
            "embedding_path",
            "model_name",
            "num_frames",
            "pooling",
            "status",
            "message",
        ])

        def write_manifest_row(
            video_path_rel: str,
            embedding_type: str,
            embedding_path: str,
            model_name: str,
            num_frames: int,
            pooling: str,
            status: str,
            message: str,
        ) -> None:
            writer.writerow([
                video_path_rel,
                embedding_type,
                embedding_path,
                model_name,
                num_frames,
                pooling,
                status,
                message,
            ])
            f.flush()
            print(
                f"[MANIFEST] video={video_path_rel} | type={embedding_type} | "
                f"status={status} | message={message}"
            )

        for video_path in tqdm(videos, desc="Extracting embeddings"):
            try:
                frames = sample_frames(video_path, num_frames=args.num_frames)
            except Exception as e:
                message = f"frame_sampling_failed: {e}"
                for embedding_type in extractors.keys():
                    write_manifest_row(
                        video_path_rel=str(video_path.relative_to(video_root)),
                        embedding_type=embedding_type,
                        embedding_path="",
                        model_name=extractors[embedding_type].model_name,
                        num_frames=args.num_frames,
                        pooling=extractors[embedding_type].pooling,
                        status="error",
                        message=message,
                    )
                print(f"[ERROR] {video_path}: {message}", file=sys.stderr)
                raise RuntimeError(f"Stopped due to sampling error on {video_path}: {e}") from e

            for embedding_type, extractor in extractors.items():
                try:
                    embedding = extractor.encode_video(frames)
                    emb_path = save_embedding(
                        output_root,
                        video_root,
                        video_path,
                        embedding,
                        embedding_type=embedding_type,
                    )
                    write_manifest_row(
                        video_path_rel=str(video_path.relative_to(video_root)),
                        embedding_type=embedding_type,
                        embedding_path=str(emb_path.relative_to(output_root)),
                        model_name=extractors[embedding_type].model_name,
                        num_frames=args.num_frames,
                        pooling=extractors[embedding_type].pooling,
                        status="ok",
                        message="",
                    )
                except Exception as e:
                    write_manifest_row(
                        video_path_rel=str(video_path.relative_to(video_root)),
                        embedding_type=embedding_type,
                        embedding_path="",
                        model_name=extractors[embedding_type].model_name,
                        num_frames=args.num_frames,
                        pooling=extractors[embedding_type].pooling,
                        status="error",
                        message=str(e),
                    )
                    print(
                        f"[ERROR] {video_path} ({embedding_type} embedding): {e}",
                        file=sys.stderr,
                    )
                    raise RuntimeError(
                        f"Stopped due to embedding error on {video_path} [{embedding_type}]: {e}"
                    ) from e

    print(f"[Done] Videos: {len(videos)} | Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract video-level and/or frame-averaged embeddings from video clips."
    )
    parser.add_argument("--video_root", type=str, required=True, help="Root directory containing video files.")
    parser.add_argument("--output_root", type=str, required=True, help="Directory to save embeddings and manifest.")
    parser.add_argument(
        "--embedding_mode",
        type=str,
        default="both",
        choices=["video", "frame", "both"],
        help="Which embeddings to extract: video-level, frame-averaged, or both.",
    )
    parser.add_argument(
        "--video_model_name",
        type=str,
        default="microsoft/xclip-base-patch32",
        help="Video model name for video-level embeddings.",
    )
    parser.add_argument(
        "--frame_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP image model name for frame-averaged embeddings.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames per video (default: 16).",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")

    run(parser.parse_args())
