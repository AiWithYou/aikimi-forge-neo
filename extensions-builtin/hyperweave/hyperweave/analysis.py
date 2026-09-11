"""Source analysis and local-only face detector providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .color import image_to_linear_rgb, luminance
from .config import ContentProfile, HyperWeaveConfig
from .frequency import StructureMapBuilder, StructureMaps, gaussian_blur


logger = logging.getLogger("hyperweave")


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: dict[str, tuple[float, float]] | None
    mask: np.ndarray | None
    detector_name: str
    source_resolution: tuple[int, int]
    original_bbox_size: tuple[float, float]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class FaceDetectorProvider(ABC):
    provider_name = "Unavailable"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def detect(
        self, image: Image.Image, config: HyperWeaveConfig
    ) -> list[FaceDetection]: ...


class ManualFaceDetector(FaceDetectorProvider):
    provider_name = "Manual ROI"

    def available(self) -> bool:
        return True

    def detect(
        self, image: Image.Image, config: HyperWeaveConfig
    ) -> list[FaceDetection]:
        if config.manual_face_mask is None:
            return []
        mask = mask_to_array(
            config.manual_face_mask,
            image.size,
            channel=config.mask_channel,
        )
        binary = (mask >= 0.25).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        detections: list[FaceDetection] = []
        for label in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[label].tolist()
            )
            if area < 16 or min(width, height) < config.minimum_face_size:
                continue
            component = np.where(labels == label, mask, 0.0).astype(np.float32)
            detections.append(
                FaceDetection(
                    bbox=(float(x), float(y), float(x + width), float(y + height)),
                    confidence=float(np.max(component)),
                    landmarks=None,
                    mask=component,
                    detector_name=self.provider_name,
                    source_resolution=image.size,
                    original_bbox_size=(float(width), float(height)),
                )
            )
        detections.sort(key=lambda item: item.width * item.height, reverse=True)
        return detections[: config.maximum_face_count]


class OpenCVHaarFaceDetector(FaceDetectorProvider):
    provider_name = "OpenCV Haar (photo fallback)"

    def __init__(self, model_path: str = ""):
        default = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.model_path = Path(model_path) if model_path else default
        self._classifier: cv2.CascadeClassifier | None = None

    def available(self) -> bool:
        return self.model_path.is_file()

    def detect(
        self, image: Image.Image, config: HyperWeaveConfig
    ) -> list[FaceDetection]:
        if not self.available():
            return []
        if self._classifier is None:
            self._classifier = cv2.CascadeClassifier(str(self.model_path))
        if self._classifier.empty():
            return []
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(config.minimum_face_size, config.minimum_face_size),
        )
        detections = [
            FaceDetection(
                bbox=(float(x), float(y), float(x + width), float(y + height)),
                confidence=0.55,
                landmarks=None,
                mask=None,
                detector_name=self.provider_name,
                source_resolution=image.size,
                original_bbox_size=(float(width), float(height)),
            )
            for x, y, width, height in boxes
        ]
        detections.sort(key=lambda item: item.width * item.height, reverse=True)
        return detections[: config.maximum_face_count]


class IdentityConditioner:
    """Explicit no-download identity-provider boundary."""

    provider_name = "None"

    def available(self) -> bool:
        return False

    def validate_reference(self, image: Image.Image | None) -> tuple[bool, str]:
        if image is None:
            return True, "No identity reference supplied."
        return False, "参照条件providerなし; reference is not applied to generation."

    def prepare(self, image: Image.Image) -> None:
        raise RuntimeError("No identity reference provider is available.")

    def apply_to_processing(self, processing: object) -> None:
        raise RuntimeError("No identity reference provider is available.")

    def score_candidate(self, candidate: Image.Image) -> float:
        return 0.0


class StructureConditioner:
    """Boundary for existing ControlNet/lineart/tile providers."""

    provider_name = "None"

    def available(self) -> bool:
        return False

    def apply_to_processing(self, processing: object, pass_name: str) -> None:
        raise RuntimeError("No stable structure-conditioner API was selected.")


@dataclass
class SourceAnalysis:
    source_size: tuple[int, int]
    source_linear_rgb: np.ndarray
    source_alpha: np.ndarray | None
    source_luminance: np.ndarray
    structure_maps: StructureMaps
    multiscale_edges: list[np.ndarray]
    line_orientation: tuple[np.ndarray, np.ndarray]
    local_coherence: np.ndarray
    flatness_map: np.ndarray
    texture_map: np.ndarray
    face_detections: list[FaceDetection]
    head_regions: list[tuple[float, float, float, float]]
    manual_protection: np.ndarray
    manual_boost: np.ndarray
    content_profile: ContentProfile
    detector_provider: str
    messages: list[str] = field(default_factory=list)


def mask_to_array(
    mask: Image.Image | None,
    size: tuple[int, int],
    *,
    channel: str = "Luminance",
) -> np.ndarray:
    if mask is None:
        return np.zeros((size[1], size[0]), dtype=np.float32)
    resized = mask.resize(size, Image.Resampling.BILINEAR).convert("RGBA")
    rgba = np.asarray(resized, dtype=np.float32) / 255.0
    luminance_value = (
        0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1] + 0.0722 * rgba[..., 2]
    )
    if channel == "Alpha":
        return rgba[..., 3].astype(np.float32)
    return (luminance_value * rgba[..., 3]).astype(np.float32)


def infer_content_profile(image: Image.Image) -> ContentProfile:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8), 80, 180
    )
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    edge_density = float(np.mean(edges > 0))
    saturation_p75 = float(np.percentile(saturation, 75.0))
    flatness = float(
        np.mean(
            np.sqrt(
                np.maximum(
                    0.0,
                    gaussian_blur(gray * gray, 2.0)
                    - gaussian_blur(gray, 2.0) ** 2,
                )
            )
            < 0.025
        )
    )
    if edge_density > 0.09 or saturation_p75 > 0.42 or flatness > 0.55:
        return ContentProfile.ILLUSTRATION
    if edge_density < 0.055 and saturation_p75 < 0.34 and flatness < 0.38:
        return ContentProfile.PHOTO
    # Uncertain auto classification intentionally stays illustration-biased.
    return ContentProfile.ILLUSTRATION


def _head_region(
    detection: FaceDetection, size: tuple[int, int]
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = detection.bbox
    width = x1 - x0
    height = y1 - y0
    return (
        max(0.0, x0 - 0.65 * width),
        max(0.0, y0 - 1.10 * height),
        min(float(size[0]), x1 + 0.65 * width),
        min(float(size[1]), y1 + 0.55 * height),
    )


class SourceAnalyzer:
    def analyze(
        self, image: Image.Image, config: HyperWeaveConfig
    ) -> SourceAnalysis:
        rgb, alpha = image_to_linear_rgb(image)
        y = luminance(rgb)
        protection = mask_to_array(
            config.protection_mask, image.size, channel=config.mask_channel
        )
        boost = mask_to_array(
            config.boost_mask, image.size, channel=config.mask_channel
        )
        structure = StructureMapBuilder().build(rgb, protection)
        multiscale_edges: list[np.ndarray] = []
        for sigma in (1.0, 2.0, 4.0):
            smooth = gaussian_blur(y, sigma)
            gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
            multiscale_edges.append(cv2.magnitude(gx, gy))

        profile = (
            infer_content_profile(image)
            if config.content_profile == ContentProfile.AUTO
            else ContentProfile(config.content_profile)
        )
        messages: list[str] = []
        manual_provider = ManualFaceDetector()
        detections = manual_provider.detect(image, config)
        provider_name = manual_provider.provider_name if detections else "Manual ROI only"

        requested = config.detector_provider
        may_use_photo = profile == ContentProfile.PHOTO
        if not detections and requested in (
            "Auto (local only)",
            "OpenCV Haar (photo only)",
        ):
            if may_use_photo:
                haar = OpenCVHaarFaceDetector(config.detector_model_path)
                if haar.available():
                    detections = haar.detect(image, config)
                    provider_name = haar.provider_name
                else:
                    messages.append("OpenCV Haar detector file is unavailable.")
            else:
                messages.append(
                    "Illustration/uncertain profile: photo Haar detections were not "
                    "adopted even when explicitly selected; use Manual Face Core Mask."
                )
        if not detections:
            messages.append("No face ROI detected; global processing will continue.")
        if config.identity_reference is not None:
            ok, message = IdentityConditioner().validate_reference(
                config.identity_reference
            )
            if not ok:
                messages.append(message)
        if config.structure_conditioner != "None":
            messages.append(
                f"Structure conditioner '{config.structure_conditioner}' is not "
                "forwarded without a stable provider API."
            )
        return SourceAnalysis(
            source_size=image.size,
            source_linear_rgb=rgb,
            source_alpha=alpha,
            source_luminance=y,
            structure_maps=structure,
            multiscale_edges=multiscale_edges,
            line_orientation=(
                structure.orientation_x,
                structure.orientation_y,
            ),
            local_coherence=structure.coherence,
            flatness_map=structure.flatness,
            texture_map=structure.texture,
            face_detections=detections,
            head_regions=[_head_region(item, image.size) for item in detections],
            manual_protection=protection,
            manual_boost=boost,
            content_profile=profile,
            detector_provider=provider_name,
            messages=messages,
        )
