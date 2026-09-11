from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TextIO

SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_HEADROOM_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_RETRIES = 3
MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


class SetupError(RuntimeError):
    """An actionable setup failure that is safe to show to a local user."""

    code = "setup_error"


class ManifestError(SetupError):
    code = "invalid_manifest"


class IntegrityError(SetupError):
    code = "integrity_error"


class DownloadError(SetupError):
    code = "download_error"


class DiskSpaceError(SetupError):
    code = "insufficient_disk_space"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    relative_path: str
    url: str
    size: int
    sha256: str
    license_url: str
    header_markers: tuple[str, ...] = ()
    marker_counts: tuple[tuple[str, int], ...] = ()
    temporary: bool = False


@dataclass(frozen=True)
class GeneratedArtifactSpec:
    relative_path: str
    size: int
    header_markers: tuple[str, ...]
    marker_counts: tuple[tuple[str, int], ...]
    sidecar_relative_path: str


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    artifacts: tuple[ArtifactSpec, ...]
    licenses: tuple[str, ...]
    legacy_peak_bytes: int
    generated: GeneratedArtifactSpec | None = None
    temporary_artifact_ids: frozenset[str] = field(default_factory=frozenset)
    runtime_revision: str | None = None
    runtime_revision_path: str | None = None

    @property
    def permanent_artifacts(self) -> tuple[ArtifactSpec, ...]:
        return tuple(item for item in self.artifacts if not item.temporary)


@dataclass(frozen=True)
class ArtifactStatus:
    artifact_id: str
    path: str
    state: str
    detail: str
    bytes_present: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "state": self.state,
            "detail": self.detail,
            "bytes_present": self.bytes_present,
        }


SENSENOVA_SOURCE_REVISION = "e6dfd45762eb46f805067fe079c14bcb643ccccd"
SENSENOVA_RUNTIME_FILES: tuple[tuple[str, str, int], ...] = (
    ("LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11357),
    (
        "SenseNova-U1.5-8B-MoT/added_tokens.json",
        "d0ff3acec259fabfafc1ffa67638aeaf58203e5e604648fb44f072e4efe040c4",
        8005,
    ),
    ("SenseNova-U1.5-8B-MoT/config.json", "6497591f64cb0dd6917fbb10c0cd13024e5817179a9aa3700998eb137a553d6b", 2631),
    ("SenseNova-U1.5-8B-MoT/merges.txt", "455e0caaa06abffc663e9282dfe71dde07fd1991eaf24146bf08793c4dba4497", 1670344),
    (
        "SenseNova-U1.5-8B-MoT/special_tokens_map.json",
        "529306ff26be5cf190b4d96781e63c7dccd03ef0a39f87c0f1289d2d5a67a02f",
        39951,
    ),
    (
        "SenseNova-U1.5-8B-MoT/tokenizer_config.json",
        "7433b95cec590c7d687259e81bca1bc4630ff39773dbf7f30f7df27a99748077",
        63546,
    ),
    ("SenseNova-U1.5-8B-MoT/vocab.json", "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d", 3383407),
    ("SenseNova/convrot_loader.py", "b6fdcfc50a1820f91ded2bde1dedab9de81fb346eb6cde28740720c0a55e3947", 60497),
    (
        "SenseNova/examples/editing/data/samples_reasoning.jsonl",
        "35821e3edc65ac34ed79cc7aa0300313f719f46cd66300d04083e1f1f1a4fa92",
        1242,
    ),
    (
        "SenseNova/examples/editing/data/samples.jsonl",
        "ced63ef279b3889b2c49a750da8d8e2a0001f7829a0d43ffcc0503279a99a065",
        937,
    ),
    (
        "SenseNova/examples/editing/inference.py",
        "bc001d58c70ab72fb084a91748322e055a843736774e528a611359d002da5d6f",
        9002,
    ),
    ("SenseNova/examples/utils.py", "1dc693564e76fd65859e4fd3137756c56f7bb3fd3d724faf2ecac2ee2c6cbe0b", 936),
    ("SenseNova/fp8_scaled_loader.py", "56fb1a66c45cef272d3833ab8eb97078a076f688008a4caed1e9b627dcb7dcc9", 20589),
    ("SenseNova/layer_streaming.py", "29a774564ad9176b61b7ae09e1cde8afd3c4921fbd42a825790615469d2092e5", 22886),
    ("SenseNova/lora_runtime.py", "c5bd4ae3d0e03e939a67034062a0e6b94af9bf5095d48b66fcdb5bbea5c3aca3", 6107),
    (
        "SenseNova/src/sensenova_u1/__init__.py",
        "ea212423e96e0811b36d961dc3721600bb905d2febf433933fecfb42392b8ba3",
        2185,
    ),
    (
        "SenseNova/src/sensenova_u1/models/__init__.py",
        "855ca4379592a6ede65421e9dbb7cfd2c0c87b6451c5868cba549f13ffdfc8dd",
        133,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/__init__.py",
        "039dd16ec24218c02a1ab5905723d8fbc8f62fd350e8614cb6cf25d1254be8fe",
        1497,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/configuration_neo_chat.py",
        "9bba1cc126cd17996006c28db3d1b2ceefd565270da7ac68f3afb30c659acc72",
        7599,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/configuration_neo_vit.py",
        "17d024a39e4b0c320b2078ebb14065907b015fa8068e812760f365381e63428f",
        1836,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/conversation.py",
        "0afd416e8875789404745ab9f05445ce2f9492a7da7b34c67436424c84f68435",
        15889,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/modeling_fm_modules.py",
        "3a492ca09604b56a8bb68c3deee0e4cbb07d40a2acb547bb40b7c6e91065e45d",
        21015,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/modeling_neo_chat.py",
        "ceda42402a49a93fa3669889a036eabe5f1a2ee108715ef24ca13174398df64f",
        97433,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/modeling_neo_vit.py",
        "98cadea49f6bde097b3644b6fa2ad908e0ff16dc47b2b7b6fcd9984b6419300d",
        8889,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/modeling_qwen3_moe.py",
        "ed3d295b33e1fcb57b0305476ee7314a0806a2934776efdd91306606f1cd148d",
        22675,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/modeling_qwen3.py",
        "3b4f1a73ca42d0d1837e83d7fb8dec424912224e584e13f80f2c46d1be969f6d",
        56946,
    ),
    (
        "SenseNova/src/sensenova_u1/models/neo_unify/utils.py",
        "6ba5efaa8bdf9a33fd3ab63767f0c8697906897ab25bacc4c3dd88926c4da003",
        6059,
    ),
)


def _sensenova_runtime_artifacts() -> tuple[ArtifactSpec, ...]:
    raw_root = f"https://raw.githubusercontent.com/starsFriday/ComfyUI-SenseNova/{SENSENOVA_SOURCE_REVISION}"
    license_url = f"https://github.com/starsFriday/ComfyUI-SenseNova/blob/{SENSENOVA_SOURCE_REVISION}/LICENSE"
    return tuple(
        ArtifactSpec(
            artifact_id=f"sensenova-runtime:{path}",
            relative_path=f"models/SenseNova-U1/runtime-final/{path}",
            url=f"{raw_root}/{urllib.parse.quote(path, safe='/')}",
            size=size,
            sha256=sha256,
            license_url=license_url,
        )
        for path, sha256, size in SENSENOVA_RUNTIME_FILES
    )


KREA_REVISION = "8038ce89b91b042141541ad0fa51b985ca262c5f"
ANIMA_REVISION = "3ef641256377dc4e7efbf35d426ca31c1fe5180b"
ANIMA_COMMON_REVISION = "f973fc41ec7545364ac9776c2440285f43ff2a30"
SENSENOVA_MODEL_REVISION = "57de22ad4e2fc24c77f56dfe45dbb87a60dfebee"
SENSENOVA_LORA_REVISION = "e909f4636d119d65fe4cba8770c19daff2ac102e"


PROFILES: Mapping[str, ProfileSpec] = {
    "krea2": ProfileSpec(
        name="krea2",
        description="Krea2 Turbo INT8 ConvRot, Qwen3-VL encoder, and Qwen Image VAE",
        artifacts=(
            ArtifactSpec(
                "krea2-checkpoint",
                "models/Stable-diffusion/krea2_turbo_int8_convrot.safetensors",
                f"https://huggingface.co/Comfy-Org/Krea-2/resolve/{KREA_REVISION}/diffusion_models/krea2_turbo_int8_convrot.safetensors",
                13_492_686_496,
                "8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1",
                f"https://huggingface.co/Comfy-Org/Krea-2/blob/{KREA_REVISION}/LICENSE.pdf",
                ("blocks.0.attn.gate.weight_scale", "blocks.0.attn.gate.comfy_quant"),
            ),
            ArtifactSpec(
                "krea2-qwen3vl",
                "models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors",
                f"https://huggingface.co/Comfy-Org/Krea-2/resolve/{KREA_REVISION}/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                5_242_467_968,
                "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094",
                "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/LICENSE",
                ("model.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight"),
            ),
            ArtifactSpec(
                "qwen-image-vae",
                "models/VAE/qwen_image_vae.safetensors",
                f"https://huggingface.co/Comfy-Org/Krea-2/resolve/{KREA_REVISION}/vae/qwen_image_vae.safetensors",
                253_806_246,
                "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
                "https://huggingface.co/Qwen/Qwen-Image/blob/main/LICENSE",
                ("conv1.weight", "decoder.conv1.weight"),
            ),
        ),
        licenses=(
            f"https://huggingface.co/Comfy-Org/Krea-2/blob/{KREA_REVISION}/LICENSE.pdf",
            "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/LICENSE",
            "https://huggingface.co/Qwen/Qwen-Image/blob/main/LICENSE",
        ),
        legacy_peak_bytes=18_988_960_710,
    ),
    "anima38": ProfileSpec(
        name="anima38",
        description="Anima 3.8B v1.1 INT8 ConvRot with Semantic Connector v2 and Qwen3.5",
        artifacts=(
            ArtifactSpec(
                "anima38-qwen35",
                "models/text_encoder/qwen35_4b.safetensors",
                f"https://huggingface.co/lylogummy/Anima-3.8B/resolve/{ANIMA_REVISION}/text_encoders/qwen35_4b.safetensors",
                4_779_016_600,
                "ea289be7c916726d09953c7db9971c82b280e694b5d7c47f8ad9ffad6acb54ba",
                "https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
                ("embed_tokens.weight", "layers.31.input_layernorm.weight"),
            ),
            ArtifactSpec(
                "anima-native-qwen",
                "models/text_encoder/qwen_3_06b_base.safetensors",
                f"https://huggingface.co/circlestone-labs/Anima/resolve/{ANIMA_COMMON_REVISION}/split_files/text_encoders/qwen_3_06b_base.safetensors",
                1_192_135_096,
                "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
                f"https://huggingface.co/circlestone-labs/Anima/blob/{ANIMA_COMMON_REVISION}/LICENSE.md",
                ("model.embed_tokens.weight",),
            ),
            ArtifactSpec(
                "anima-qwen-image-vae",
                "models/VAE/qwen_image_vae.safetensors",
                f"https://huggingface.co/circlestone-labs/Anima/resolve/{ANIMA_COMMON_REVISION}/split_files/vae/qwen_image_vae.safetensors",
                253_806_246,
                "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
                f"https://huggingface.co/circlestone-labs/Anima/blob/{ANIMA_COMMON_REVISION}/LICENSE.md",
                ("decoder.conv1.weight",),
            ),
            ArtifactSpec(
                "anima38-bf16-source",
                "tmp/anima38-v11-conversion/Anima-3.8B-v1.1.safetensors",
                f"https://huggingface.co/lylogummy/Anima-3.8B/resolve/{ANIMA_REVISION}/difussion_models/Anima-3.8B-v1.1.safetensors",
                8_809_227_318,
                "4a458d26b21efa350073422f756d521b4397d9ca5964da4dc6bd9ae258a29629",
                f"https://huggingface.co/lylogummy/Anima-3.8B/tree/{ANIMA_REVISION}#licenses",
                (
                    "anima_3_8b_semantic_connector_v2_bundle",
                    "net.anima_v2_connector.semantic_resampler.query_tokens",
                    "net.blocks.51.mlp.layer2.weight",
                ),
                temporary=True,
            ),
        ),
        licenses=(
            f"https://huggingface.co/lylogummy/Anima-3.8B/tree/{ANIMA_REVISION}#licenses",
            f"https://huggingface.co/circlestone-labs/Anima/blob/{ANIMA_COMMON_REVISION}/LICENSE.md",
            "https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        ),
        legacy_peak_bytes=19_131_608_492,
        generated=GeneratedArtifactSpec(
            "models/Stable-diffusion/Anima-3.8B-v1.1-int8-convrot.safetensors",
            5_543_364_574,
            (
                "anima38_v11_main_attention_mlp_v1",
                "net.anima_v2_connector.semantic_resampler.query_tokens",
                "net.blocks.0.self_attn.q_proj.weight_scale",
                "net.blocks.51.mlp.layer2.comfy_quant",
            ),
            (('.comfy_quant"', 520),),
            "models/Stable-diffusion/Anima-3.8B-v1.1-int8-convrot.safetensors.sha256",
        ),
        temporary_artifact_ids=frozenset({"anima38-bf16-source"}),
    ),
    "sensenova": ProfileSpec(
        name="sensenova",
        description="SenseNova U1.5 runtime, INT8 ConvRot checkpoint, and official 8-step LoRA",
        artifacts=(
            *_sensenova_runtime_artifacts(),
            ArtifactSpec(
                "sensenova-checkpoint",
                "models/SenseNova-U1/SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors",
                f"https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8/resolve/{SENSENOVA_MODEL_REVISION}/SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors",
                17_734_813_848,
                "cf6ed9ee3be516612b7fe083edfc7c9dd5d059cc759e300d2cf1f2726c0d250e",
                f"https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8/tree/{SENSENOVA_MODEL_REVISION}",
                (".comfy_quant", "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"),
                (('.comfy_quant"', 588),),
            ),
            ArtifactSpec(
                "sensenova-8step-lora",
                "models/SenseNova-U1/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors",
                f"https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/resolve/{SENSENOVA_LORA_REVISION}/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors",
                814_867_236,
                "3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708",
                f"https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/tree/{SENSENOVA_LORA_REVISION}",
                ('"tensor_kind":"neo_hf_lora"',),
                (
                    ('.lora_down.weight"', 294),
                    ('.lora_up.weight"', 294),
                    ('.alpha"', 294),
                ),
            ),
        ),
        licenses=(
            f"https://github.com/starsFriday/ComfyUI-SenseNova/blob/{SENSENOVA_SOURCE_REVISION}/LICENSE",
            f"https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs/tree/{SENSENOVA_LORA_REVISION}",
            f"https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8/tree/{SENSENOVA_MODEL_REVISION}",
        ),
        legacy_peak_bytes=36_284_494_932,
        runtime_revision=SENSENOVA_SOURCE_REVISION,
        runtime_revision_path="models/SenseNova-U1/runtime-final/.sensenova_runtime_revision",
    ),
}


def format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MiB"
    return f"{value} bytes"


def _relative_parts(value: str) -> tuple[str, ...]:
    normalized = value.replace("\\", "/")
    if not normalized or normalized.startswith(("/", "//")):
        raise ManifestError(f"Artifact path must be relative: {value!r}")
    if re.match(r"^[A-Za-z]:", normalized):
        raise ManifestError(f"Drive-qualified artifact path is not allowed: {value!r}")
    raw_parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ManifestError(f"Artifact path contains an unsafe component: {value!r}")
    path = PurePosixPath(normalized)
    return tuple(path.parts)


def _safe_public_url(value: str, *, allow_test_http: bool = False, allow_fragment: bool = False) -> str:
    parsed = urllib.parse.urlsplit(value)
    allowed_schemes = {"https"} | ({"http"} if allow_test_http else set())
    if parsed.scheme.lower() not in allowed_schemes:
        raise ManifestError("Artifact URLs must use HTTPS")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ManifestError("Artifact URL contains userinfo or has no hostname")
    if parsed.query or (parsed.fragment and not allow_fragment):
        raise ManifestError("Artifact URL query strings and fragments are not allowed")
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, "", ""))


def _display_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    host = parsed.hostname or "invalid-host"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", parsed.fragment))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header_text(path: Path) -> str:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise IntegrityError("SafeTensors header length is missing")
        header_length = int.from_bytes(length_bytes, "little", signed=False)
        if not 2 < header_length <= MAX_SAFETENSORS_HEADER_BYTES:
            raise IntegrityError("SafeTensors header length is outside the allowed range")
        if header_length > path.stat().st_size - 8:
            raise IntegrityError("SafeTensors header is truncated")
        header = stream.read(header_length)
    try:
        text = header.decode("utf-8")
        json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("SafeTensors header is not valid UTF-8 JSON") from error
    return text


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, allow_test_http: bool) -> None:
        super().__init__()
        self.allow_test_http = allow_test_http

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        parsed = urllib.parse.urlsplit(newurl)
        allowed_schemes = {"https"} | ({"http"} if self.allow_test_http else set())
        if parsed.scheme.lower() not in allowed_schemes:
            raise DownloadError("Download redirect used a disallowed URL scheme")
        if parsed.username is not None or parsed.password is not None:
            raise DownloadError("Download redirect contained URL userinfo")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
            redirected.remove_header("Cookie")
        return redirected


class Installer:
    def __init__(
        self,
        root: Path,
        profiles: Mapping[str, ProfileSpec] = PROFILES,
        *,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
        json_mode: bool = False,
        allow_test_http: bool = False,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        self.root = root.resolve()
        self.profiles = dict(profiles)
        self.stdout = stdout
        self.stderr = stderr
        self.json_mode = json_mode
        self.allow_test_http = allow_test_http
        self.disk_usage = disk_usage
        self.opener = opener or urllib.request.build_opener(SafeRedirectHandler(allow_test_http=allow_test_http))
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        destinations: set[str] = set()
        artifact_ids: set[str] = set()
        for profile_name, profile in self.profiles.items():
            if profile_name != profile.name:
                raise ManifestError(f"Profile key does not match profile name: {profile_name}")
            for artifact in profile.artifacts:
                if artifact.artifact_id in artifact_ids:
                    raise ManifestError(f"Duplicate artifact ID: {artifact.artifact_id}")
                artifact_ids.add(artifact.artifact_id)
                canonical_path = "/".join(_relative_parts(artifact.relative_path)).casefold()
                if canonical_path in destinations:
                    # The shared Qwen Image VAE is intentionally represented by two
                    # source profiles but must have identical integrity metadata.
                    matching = [
                        item
                        for other in self.profiles.values()
                        for item in other.artifacts
                        if "/".join(_relative_parts(item.relative_path)).casefold() == canonical_path
                    ]
                    if any(item.size != artifact.size or item.sha256 != artifact.sha256 for item in matching):
                        raise ManifestError(f"Conflicting metadata for shared destination: {artifact.relative_path}")
                destinations.add(canonical_path)
                _safe_public_url(artifact.url, allow_test_http=self.allow_test_http)
                _safe_public_url(
                    artifact.license_url,
                    allow_test_http=self.allow_test_http,
                    allow_fragment=True,
                )
                if artifact.size <= 0:
                    raise ManifestError(f"Artifact size must be positive: {artifact.artifact_id}")
                if not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256):
                    raise ManifestError(f"Invalid SHA-256: {artifact.artifact_id}")
            if profile.generated is not None:
                _relative_parts(profile.generated.relative_path)
                _relative_parts(profile.generated.sidecar_relative_path)
            if profile.runtime_revision_path is not None:
                _relative_parts(profile.runtime_revision_path)

    def _target(self, relative_path: str) -> Path:
        parts = _relative_parts(relative_path)
        lexical = self.root.joinpath(*parts)
        resolved = lexical.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise SetupError(f"Managed path escapes the repository root: {relative_path}") from error
        return lexical

    def _ensure_parent(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(self.root)
        except ValueError as error:
            raise SetupError("Managed directory escaped the repository root") from error

    def _log(self, message: str) -> None:
        print(message, file=self.stderr if self.json_mode else self.stdout, flush=True)

    @contextmanager
    def _mutation_lock(self):
        """Hold one nonblocking setup lock across every filesystem mutation."""

        lock_path = self._target("tmp/aikimi-setup/setup.lock")
        self._ensure_parent(lock_path)
        with lock_path.open("a+b") as lock_file:
            if lock_file.seek(0, os.SEEK_END) == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise SetupError("Another Aikimi model setup process is already running") from error
            try:
                yield
            finally:
                lock_file.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _verify_file(self, artifact: ArtifactSpec, path: Path) -> None:
        if path.is_symlink():
            raise IntegrityError(f"Managed artifact is a symbolic link: {artifact.artifact_id}")
        actual_size = path.stat().st_size
        if actual_size != artifact.size:
            raise IntegrityError(
                f"Size mismatch for {artifact.artifact_id}: expected {artifact.size}, got {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != artifact.sha256:
            raise IntegrityError(f"SHA-256 mismatch for {artifact.artifact_id}")
        if artifact.header_markers or artifact.marker_counts:
            header = _safetensors_header_text(path)
            for marker in artifact.header_markers:
                if marker not in header:
                    raise IntegrityError(f"SafeTensors marker is missing for {artifact.artifact_id}")
            for marker, expected_count in artifact.marker_counts:
                actual_count = header.count(marker)
                if actual_count != expected_count:
                    raise IntegrityError(
                        f"SafeTensors marker count mismatch for {artifact.artifact_id}: "
                        f"expected {expected_count}, got {actual_count}"
                    )

    def _artifact_status(self, artifact: ArtifactSpec) -> ArtifactStatus:
        target = self._target(artifact.relative_path)
        partial = Path(f"{target}.part")
        if target.exists():
            if not target.is_file() and not target.is_symlink():
                return ArtifactStatus(
                    artifact.artifact_id,
                    artifact.relative_path,
                    "invalid",
                    "managed path is not a regular file",
                )
            try:
                self._verify_file(artifact, target)
            except (OSError, SetupError) as error:
                return ArtifactStatus(
                    artifact.artifact_id,
                    artifact.relative_path,
                    "invalid",
                    str(error),
                    target.lstat().st_size,
                )
            return ArtifactStatus(
                artifact.artifact_id,
                artifact.relative_path,
                "ready",
                "size and SHA-256 verified",
                artifact.size,
            )
        if partial.exists() and partial.is_file():
            return ArtifactStatus(
                artifact.artifact_id,
                artifact.relative_path,
                "partial",
                "resumable partial file exists",
                partial.stat().st_size,
            )
        return ArtifactStatus(
            artifact.artifact_id,
            artifact.relative_path,
            "missing",
            "artifact is not installed",
        )

    def _verify_generated(self, profile: ProfileSpec) -> ArtifactStatus | None:
        generated = profile.generated
        if generated is None:
            return None
        target = self._target(generated.relative_path)
        sidecar = self._target(generated.sidecar_relative_path)
        if not target.exists():
            return ArtifactStatus(
                f"{profile.name}-generated",
                generated.relative_path,
                "missing",
                "converted artifact is not installed",
            )
        try:
            if target.is_symlink() or not target.is_file():
                raise IntegrityError("Converted artifact is not a regular managed file")
            if target.stat().st_size != generated.size:
                raise IntegrityError("Converted artifact size mismatch")
            header = _safetensors_header_text(target)
            for marker in generated.header_markers:
                if marker not in header:
                    raise IntegrityError("Converted artifact marker is missing")
            for marker, expected_count in generated.marker_counts:
                if header.count(marker) != expected_count:
                    raise IntegrityError("Converted artifact layer count mismatch")
            if not sidecar.is_file() or sidecar.is_symlink():
                raise IntegrityError("Converted artifact checksum sidecar is missing")
            record = sidecar.read_text(encoding="ascii").strip()
            match = re.fullmatch(rf"([0-9a-f]{{64}})  {re.escape(target.name)}", record)
            if match is None or _sha256(target) != match.group(1):
                raise IntegrityError("Converted artifact checksum does not match")
        except (OSError, SetupError, UnicodeError) as error:
            return ArtifactStatus(
                f"{profile.name}-generated",
                generated.relative_path,
                "invalid",
                str(error),
                target.lstat().st_size,
            )
        return ArtifactStatus(
            f"{profile.name}-generated",
            generated.relative_path,
            "ready",
            "converted artifact and sidecar verified",
            generated.size,
        )

    def _runtime_revision_status(self, profile: ProfileSpec) -> ArtifactStatus | None:
        if profile.runtime_revision is None or profile.runtime_revision_path is None:
            return None
        target = self._target(profile.runtime_revision_path)
        if not target.is_file() or target.is_symlink():
            return ArtifactStatus(
                f"{profile.name}-runtime-revision",
                profile.runtime_revision_path,
                "missing",
                "runtime revision record is missing",
            )
        try:
            value = target.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            return ArtifactStatus(
                f"{profile.name}-runtime-revision",
                profile.runtime_revision_path,
                "invalid",
                f"runtime revision record cannot be read: {error.__class__.__name__}",
            )
        if value != profile.runtime_revision:
            return ArtifactStatus(
                f"{profile.name}-runtime-revision",
                profile.runtime_revision_path,
                "invalid",
                "runtime revision does not match the pinned manifest",
            )
        return ArtifactStatus(
            f"{profile.name}-runtime-revision",
            profile.runtime_revision_path,
            "ready",
            "runtime revision verified",
            target.stat().st_size,
        )

    def verify(self, profile_names: Sequence[str]) -> dict[str, object]:
        reports = []
        overall_ok = True
        for profile_name in profile_names:
            profile = self._profile(profile_name)
            statuses = [self._artifact_status(item) for item in profile.permanent_artifacts]
            generated_status = self._verify_generated(profile)
            if generated_status is not None:
                statuses.append(generated_status)
            revision_status = self._runtime_revision_status(profile)
            if revision_status is not None:
                statuses.append(revision_status)
            ready = all(item.state == "ready" for item in statuses)
            overall_ok = overall_ok and ready
            reports.append(
                {
                    "profile": profile.name,
                    "ready": ready,
                    "artifacts": [item.as_dict() for item in statuses],
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "command": "verify",
            "ok": overall_ok,
            "profiles": reports,
        }

    def _profile(self, name: str) -> ProfileSpec:
        try:
            return self.profiles[name]
        except KeyError as error:
            raise SetupError(f"Unknown setup profile: {name}") from error

    def _profile_download_requirement(self, profile: ProfileSpec, *, generated_ready: bool) -> int:
        required = 0
        for artifact in profile.artifacts:
            if artifact.temporary and generated_ready:
                continue
            target = self._target(artifact.relative_path)
            partial = Path(f"{target}.part")
            if target.is_file() and target.stat().st_size == artifact.size:
                continue
            partial_bytes = min(partial.stat().st_size, artifact.size) if partial.is_file() else 0
            required += artifact.size - partial_bytes
        if profile.generated is not None and not generated_ready:
            generated_path = self._target(profile.generated.relative_path)
            generated_partial = Path(f"{generated_path}.part")
            partial_bytes = (
                min(generated_partial.stat().st_size, profile.generated.size) if generated_partial.is_file() else 0
            )
            required += profile.generated.size - partial_bytes
        return required

    def _needs_install_for_plan(self, artifact: ArtifactSpec) -> bool:
        """Plan from metadata only; dry-run must not hash multi-gigabyte files."""

        target = self._target(artifact.relative_path)
        return not (target.is_file() and not target.is_symlink() and target.stat().st_size == artifact.size)

    def _generated_present_for_plan(self, profile: ProfileSpec) -> bool:
        generated = profile.generated
        if generated is None:
            return False
        target = self._target(generated.relative_path)
        sidecar = self._target(generated.sidecar_relative_path)
        return (
            target.is_file()
            and not target.is_symlink()
            and target.stat().st_size == generated.size
            and sidecar.is_file()
            and not sidecar.is_symlink()
        )

    def _disk_preflight(self, profile: ProfileSpec, *, generated_ready: bool) -> int:
        required = (
            self._profile_download_requirement(profile, generated_ready=generated_ready) + DOWNLOAD_HEADROOM_BYTES
        )
        usage = self.disk_usage(self.root)
        free = int(usage.free)
        if free < required:
            raise DiskSpaceError(
                f"Profile {profile.name} needs about {format_bytes(required)} of free space; "
                f"only {format_bytes(free)} is available"
            )
        return required

    def _request(self, artifact: ArtifactSpec, offset: int):
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "Aikimi-Neo-Setup/1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        # Production artifact URLs are validated as HTTPS-only when the
        # immutable manifest is loaded. Tests may explicitly enable localhost.
        request = urllib.request.Request(  # noqa: S310
            artifact.url, headers=headers, method="GET"
        )
        return self.opener.open(request, timeout=DEFAULT_TIMEOUT_SECONDS)

    @staticmethod
    def _validate_content_range(value: str | None, offset: int, total: int) -> None:
        if value is None:
            raise DownloadError("Resume response did not include Content-Range")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
        if match is None:
            raise DownloadError("Resume response included an invalid Content-Range")
        start, end, reported_total = (int(item) for item in match.groups())
        if start != offset or end < start or reported_total != total:
            raise DownloadError("Resume response did not match the requested byte range")

    def _download(self, artifact: ArtifactSpec) -> str:
        target = self._target(artifact.relative_path)
        self._ensure_parent(target)
        partial = Path(f"{target}.part")
        if partial.exists() and (partial.is_symlink() or not partial.is_file()):
            raise IntegrityError(f"Unsafe partial path for {artifact.artifact_id}")
        if partial.is_file() and partial.stat().st_size > artifact.size:
            raise IntegrityError(f"Partial file is larger than expected for {artifact.artifact_id}; run repair")

        for attempt in range(1, DEFAULT_RETRIES + 1):
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset == artifact.size:
                break
            try:
                with self._request(artifact, offset) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    if status not in {200, 206}:
                        raise DownloadError(f"Unexpected HTTP status for {artifact.artifact_id}")
                    if status == 206:
                        self._validate_content_range(response.headers.get("Content-Range"), offset, artifact.size)
                        mode = "ab"
                        expected_response_bytes = artifact.size - offset
                    else:
                        # A server may ignore Range. Restarting the partial is safe;
                        # appending a complete 200 response would corrupt the file.
                        mode = "wb"
                        offset = 0
                        expected_response_bytes = artifact.size
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            declared = int(content_length)
                        except ValueError as error:
                            raise DownloadError(f"Invalid Content-Length for {artifact.artifact_id}") from error
                        if declared > expected_response_bytes:
                            raise DownloadError(f"Response is larger than expected for {artifact.artifact_id}")
                    with partial.open(mode) as destination:
                        written = offset
                        last_report = time.monotonic()
                        while True:
                            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > artifact.size:
                                raise DownloadError(f"Download exceeded the pinned size for {artifact.artifact_id}")
                            destination.write(chunk)
                            now = time.monotonic()
                            if now - last_report >= 1.0:
                                self._log(
                                    f"{artifact.artifact_id}: {format_bytes(written)} / {format_bytes(artifact.size)}"
                                )
                                last_report = now
                        destination.flush()
                        os.fsync(destination.fileno())
            except DownloadError:
                raise
            except (
                OSError,
                TimeoutError,
                http.client.IncompleteRead,
                urllib.error.URLError,
            ) as error:
                if attempt == DEFAULT_RETRIES:
                    raise DownloadError(
                        f"Download failed for {artifact.artifact_id} after {attempt} attempts"
                    ) from error
                self._log(f"{artifact.artifact_id}: connection interrupted; retrying ({attempt}/{DEFAULT_RETRIES})")
                continue
            if partial.stat().st_size == artifact.size:
                break

        if not partial.is_file() or partial.stat().st_size != artifact.size:
            raise DownloadError(f"Download ended before the pinned size for {artifact.artifact_id}")
        self._verify_file(artifact, partial)
        os.replace(partial, target)
        self._log(f"{artifact.artifact_id}: verified and installed")
        return artifact.relative_path

    def _write_runtime_revision(self, profile: ProfileSpec) -> None:
        if profile.runtime_revision is None or profile.runtime_revision_path is None:
            return
        target = self._target(profile.runtime_revision_path)
        self._ensure_parent(target)
        partial = Path(f"{target}.part")
        payload = f"{profile.runtime_revision}\n".encode("ascii")
        with partial.open("wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(partial, target)

    def _convert_anima(self, profile: ProfileSpec) -> None:
        generated = profile.generated
        if generated is None:
            raise SetupError("Anima conversion output is not defined")
        source_spec = next(item for item in profile.artifacts if item.artifact_id == "anima38-bf16-source")
        source = self._target(source_spec.relative_path)
        output = self._target(generated.relative_path)
        python = self.root / "venv" / "Scripts" / "python.exe"
        converter = self.root / "tools" / "convert_anima38_int8_convrot.py"
        if not python.is_file():
            raise SetupError("Forge Python is missing; start Local Safe once before converting Anima 3.8B")
        if not converter.is_file():
            raise SetupError("Anima 3.8B converter is missing")
        self._ensure_parent(output)
        command = [
            str(python),
            "-B",
            "-m",
            "tools.convert_anima38_int8_convrot",
            str(source),
            str(output),
            "--device",
            "cuda:0",
            "--group-size",
            "256",
        ]
        self._log("anima38: converting the pinned BF16 source with CUDA device 0")
        if self.json_mode:
            result = subprocess.run(  # noqa: S603 - executable and arguments are fixed managed paths
                command,
                cwd=self.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.stdout:
                print(result.stdout, file=self.stderr, end="")
        else:
            result = subprocess.run(  # noqa: S603 - executable and arguments are fixed managed paths
                command, cwd=self.root, check=False
            )
        if result.returncode != 0:
            raise SetupError("Anima 3.8B conversion failed; the BF16 source and partial output were kept for repair")
        status = self._verify_generated(profile)
        if status is None or status.state != "ready":
            raise IntegrityError("Anima 3.8B converted output failed verification")

    def install(
        self,
        profile_name: str,
        *,
        dry_run: bool,
        keep_source: bool,
        _lock: bool = True,
    ) -> dict[str, object]:
        if not dry_run and _lock:
            with self._mutation_lock():
                return self.install(
                    profile_name,
                    dry_run=False,
                    keep_source=keep_source,
                    _lock=False,
                )
        profile = self._profile(profile_name)
        if dry_run:
            generated_ready = self._generated_present_for_plan(profile)
            required_bytes = (
                self._profile_download_requirement(profile, generated_ready=generated_ready) + DOWNLOAD_HEADROOM_BYTES
            )
            planned = [
                item.relative_path
                for item in profile.artifacts
                if not (item.temporary and generated_ready) and self._needs_install_for_plan(item)
            ]
            if profile.generated is not None and not generated_ready:
                planned.append(profile.generated.relative_path)
            return {
                "schema_version": SCHEMA_VERSION,
                "command": "install",
                "profile": profile.name,
                "ok": True,
                "dry_run": True,
                "required_free_bytes": required_bytes,
                "planned_paths": planned,
                "note": "dry-run checks paths and sizes only; run verify for full SHA-256 validation",
                "licenses": list(profile.licenses),
            }

        generated_status = self._verify_generated(profile)
        generated_ready = generated_status is not None and generated_status.state == "ready"
        self._disk_preflight(profile, generated_ready=generated_ready)
        installed: list[str] = []
        skipped: list[str] = []
        for artifact in profile.artifacts:
            if artifact.temporary and generated_ready:
                continue
            status = self._artifact_status(artifact)
            if status.state == "ready":
                skipped.append(artifact.relative_path)
                continue
            if status.state == "invalid":
                raise IntegrityError(f"Existing artifact is invalid: {artifact.artifact_id}; run repair")
            installed.append(self._download(artifact))

        if profile.generated is not None and not generated_ready:
            generated_status = self._verify_generated(profile)
            if generated_status is not None and generated_status.state == "invalid":
                raise IntegrityError(f"Existing converted artifact is invalid for {profile.name}; run repair")
            self._convert_anima(profile)
            installed.append(profile.generated.relative_path)
            if not keep_source:
                for artifact in profile.artifacts:
                    if artifact.artifact_id not in profile.temporary_artifact_ids:
                        continue
                    source = self._target(artifact.relative_path)
                    if source.is_file() and not source.is_symlink():
                        source.unlink()
                        self._log(f"{artifact.artifact_id}: removed temporary conversion source")

        self._write_runtime_revision(profile)
        revision_status = self._runtime_revision_status(profile)
        if revision_status is not None and revision_status.state != "ready":
            raise IntegrityError(f"Runtime revision record failed verification: {profile.name}")
        return {
            "schema_version": SCHEMA_VERSION,
            "command": "install",
            "profile": profile.name,
            "ok": True,
            "dry_run": False,
            "installed": installed,
            "skipped": skipped,
            "licenses": list(profile.licenses),
        }

    def _quarantine(self, profile: ProfileSpec, path: Path, *, dry_run: bool) -> str:
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise SetupError("Refusing to quarantine a path outside the repository") from error
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine_relative = f"tmp/aikimi-setup/quarantine/{profile.name}/{stamp}-{path.name}"
        quarantine = self._target(quarantine_relative)
        counter = 1
        while quarantine.exists():
            quarantine_relative = f"tmp/aikimi-setup/quarantine/{profile.name}/{stamp}-{counter}-{path.name}"
            quarantine = self._target(quarantine_relative)
            counter += 1
        if not dry_run:
            self._ensure_parent(quarantine)
            os.replace(path, quarantine)
        self._log(f"quarantine: {relative} -> {quarantine_relative}")
        return quarantine_relative

    def repair(
        self,
        profile_name: str,
        *,
        dry_run: bool,
        keep_source: bool,
        _lock: bool = True,
    ) -> dict[str, object]:
        if not dry_run and _lock:
            with self._mutation_lock():
                return self.repair(
                    profile_name,
                    dry_run=False,
                    keep_source=keep_source,
                    _lock=False,
                )
        profile = self._profile(profile_name)
        quarantined: list[str] = []
        for artifact in profile.artifacts:
            target = self._target(artifact.relative_path)
            partial = Path(f"{target}.part")
            status = self._artifact_status(artifact)
            if status.state == "invalid":
                if target.is_symlink():
                    raise SetupError(f"Refusing to repair a symbolic-link target: {artifact.relative_path}")
                quarantined.append(self._quarantine(profile, target, dry_run=dry_run))
            if partial.is_file() and not partial.is_symlink():
                # Repair is an explicit request for a clean retry. Incomplete
                # files cannot be authenticated until every byte is present.
                quarantined.append(self._quarantine(profile, partial, dry_run=dry_run))

        generated_status = self._verify_generated(profile)
        if profile.generated is not None and generated_status is not None:
            generated = profile.generated
            if generated_status.state == "invalid":
                target = self._target(generated.relative_path)
                sidecar = self._target(generated.sidecar_relative_path)
                if target.is_symlink() or sidecar.is_symlink():
                    raise SetupError("Refusing to repair a symbolic-link conversion output")
                if target.exists():
                    quarantined.append(self._quarantine(profile, target, dry_run=dry_run))
                if sidecar.exists():
                    quarantined.append(self._quarantine(profile, sidecar, dry_run=dry_run))
            partial = Path(f"{self._target(generated.relative_path)}.part")
            if partial.exists():
                if partial.is_symlink() or not partial.is_file():
                    raise SetupError("Refusing to repair an unsafe conversion partial")
                quarantined.append(self._quarantine(profile, partial, dry_run=dry_run))

        revision_status = self._runtime_revision_status(profile)
        if revision_status is not None and revision_status.state == "invalid":
            revision_path = self._target(profile.runtime_revision_path or "")
            quarantined.append(self._quarantine(profile, revision_path, dry_run=dry_run))

        if dry_run:
            install_plan = self.install(
                profile.name,
                dry_run=True,
                keep_source=keep_source,
                _lock=False,
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "command": "repair",
                "profile": profile.name,
                "ok": True,
                "dry_run": True,
                "quarantined": quarantined,
                "install_plan": install_plan,
            }
        installed = self.install(
            profile.name,
            dry_run=False,
            keep_source=keep_source,
            _lock=False,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "command": "repair",
            "profile": profile.name,
            "ok": True,
            "dry_run": False,
            "quarantined": quarantined,
            "install": installed,
        }

    def list_profiles(self) -> dict[str, object]:
        profiles = []
        for profile in self.profiles.values():
            profiles.append(
                {
                    "name": profile.name,
                    "description": profile.description,
                    "artifact_count": len(profile.artifacts) + (1 if profile.generated is not None else 0),
                    "legacy_peak_bytes": profile.legacy_peak_bytes,
                    "licenses": list(profile.licenses),
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "command": "list",
            "ok": True,
            "profiles": profiles,
        }


def _print_human(report: Mapping[str, object], stdout: TextIO) -> None:
    command = report.get("command")
    if command == "list":
        for profile in report["profiles"]:  # type: ignore[index]
            print(
                f"{profile['name']}: {profile['description']} "
                f"(legacy peak {format_bytes(profile['legacy_peak_bytes'])})",
                file=stdout,
            )
            for license_url in profile["licenses"]:
                print(f"  license: {_display_url(license_url)}", file=stdout)
        return
    if command == "verify":
        for profile in report["profiles"]:  # type: ignore[index]
            print(
                f"{profile['profile']}: {'Ready' if profile['ready'] else 'Blocked'}",
                file=stdout,
            )
            for artifact in profile["artifacts"]:
                print(
                    f"  {artifact['state']}: {artifact['path']} - {artifact['detail']}",
                    file=stdout,
                )
        return
    profile = report.get("profile", "unknown")
    dry_run = " (dry-run)" if report.get("dry_run") else ""
    print(f"{command} {profile}: completed{dry_run}", file=stdout)
    for license_url in report.get("licenses", []):
        print(f"  license: {_display_url(license_url)}", file=stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install and verify pinned Aikimi Studio Neo model profiles safely.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available profiles")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON")

    verify_parser = subparsers.add_parser("verify", help="Verify installed files")
    verify_parser.add_argument("profile", nargs="?", choices=tuple(PROFILES))
    verify_parser.add_argument("--json", action="store_true", help="Emit JSON")

    for command in ("install", "repair"):
        command_parser = subparsers.add_parser(command, help=f"{command.title()} a pinned model profile")
        command_parser.add_argument("profile", choices=tuple(PROFILES))
        command_parser.add_argument("--dry-run", action="store_true", help="Plan without writing or downloading")
        command_parser.add_argument(
            "--keep-source",
            action="store_true",
            help="Keep the Anima BF16 conversion source",
        )
        command_parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    profiles: Mapping[str, ProfileSpec] = PROFILES,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_mode = bool(getattr(args, "json", False))
    installer = Installer(repository_root, profiles, json_mode=json_mode)
    try:
        if args.command == "list":
            report = installer.list_profiles()
        elif args.command == "verify":
            names = [args.profile] if args.profile else list(profiles)
            report = installer.verify(names)
        elif args.command == "install":
            report = installer.install(
                args.profile,
                dry_run=args.dry_run,
                keep_source=args.keep_source,
            )
        else:
            report = installer.repair(
                args.profile,
                dry_run=args.dry_run,
                keep_source=args.keep_source,
            )
    except SetupError as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "command": args.command,
            "ok": False,
            "error": {"code": error.code, "message": str(error)},
        }
        if json_mode:
            sys.stdout.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"ERROR [{error.code}]: {error}\n")
        return 1

    if json_mode:
        sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        _print_human(report, sys.stdout)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
