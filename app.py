from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, field_validator

from h3studio.config import ROOT as PROJECT_ROOT, path as config_path, section as config_section
from h3studio.provider import load_provider


ROOT = PROJECT_ROOT
INDEX_FILE = ROOT / "index.html"
_STORAGE = config_section("storage")


def _configured_or_legacy(key: str, legacy_name: str) -> Path:
    section_name = "local" if key == "log_file" else "storage"
    configured = config_path(section_name, key)
    legacy = ROOT / legacy_name
    # Existing installations keep working; new state is written to the organized
    # data directory once the migration command has been run.
    return legacy if legacy.exists() and not configured.exists() else configured


JOBS_FILE = _configured_or_legacy("jobs_file", ".h3_frontend_jobs.json")
BACKEND_LOG = _configured_or_legacy("log_file", ".h3_backend.log")
MODEL_STATE_FILE = _configured_or_legacy("model_state_file", ".h3_model_state.json")
QUEUE_ASSET_ROOT = _configured_or_legacy("queue_assets", ".h3_queue_assets")
LATENT_CHECKPOINT_ROOT = _configured_or_legacy("latent_checkpoints", ".h3_latent_checkpoints")
DIRECTOR_FILE = _configured_or_legacy("director_file", ".h3_director_projects.json")
DIRECTOR_ASSET_ROOT = _configured_or_legacy("director_assets", ".h3_director_assets")
# 全局素材库：跨任务、跨项目复用同一批素材，三种生成模式都能从这里导入，
# 免得每次都重新上传同一张人物设定图。
LIBRARY_FILE = _configured_or_legacy("library_file", ".h3_media_library.json")
LIBRARY_ROOT = _configured_or_legacy("library_root", ".h3_media_library")
OUTPUT_ROOT = config_path("storage", "outputs")
LEGACY_REFERENCE_IMAGE = ROOT / "74ab7025fae1545f4ceb22672cdde755.jpg"
_PROVIDER = config_section("provider")
H3_PROVIDER = load_provider()
FRONTEND_INTERNAL_URL = str(config_section("app").get("internal_url", "http://127.0.0.1:7860")).rstrip("/")
API_KEY_FILE = Path(os.environ.get("H3_API_KEY_FILE", str(_PROVIDER.get("api_key", "")))).expanduser()
H3_BASE_URL = os.environ.get("H3_BASE_URL", str(_PROVIDER.get("base_url", "http://127.0.0.1:8000"))).rstrip("/")
# 当前后端加载的是哪个 DiT 分区。两个分区（FL2VA / Ref2VA）各 62GB，同时加载会
# 让宿主内存冲到 263GB > 251GB 而被 OOM killer 杀掉，所以一次只能起一个。
# run_h3.sh 用同名环境变量选择分区，这里必须与它保持一致。
H3_PARTITION = os.environ.get("H3_PARTITION", str(config_section("local").get("partition", "ref2va"))).lower()
PARTITION_TASKS = {"ref2va": {"ref2va"}, "fl2va": {"t2va", "fl2va"}}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# ---------------------------------------------------------------------------
# MiniMax-H3 请求契约
#
# 下面每个常量都对照 vllm_omni 源码核实过，不是猜的：
#   diffusion/models/minimax_h3/pipeline_minimax_h3.py  画布/时长/参考图约束
#   diffusion/models/minimax_h3/quality_policy.py       quality 取值
#   entrypoints/openai/serving_video.py                 表单字段 -> extra_args 映射
#
# H3 pipeline 只读取这些字段：quality / fps / num_frames / width / height /
# seed / num_inference_steps / num_outputs_per_prompt，以及 extra_args 里的
# task / duration / aspect_ratio / short_edge / flow_shift / audio_flow_shift /
# generate_sound / sound_duration / force_refresh_step_{hint,policy}。
#
# 其余 OpenAPI 暴露的字段（guidance_scale、guidance_scale_2、true_cfg_scale、
# boundary_ratio、negative_prompt）属于 Wan2.2 契约，H3 pipeline 引用数为 0，
# 传了会被静默忽略，所以前端一律不暴露。
# ---------------------------------------------------------------------------
H3_FPS = 24
H3_SHORT_EDGE = 768  # pipeline 硬校验：短边必须正好是 768，其它值直接报错
H3_MAX_PIXELS = 768 * 1344
H3_FRAME_MODULUS = 17  # 帧数必须满足 17n+5
H3_FRAME_REMAINDER = 5
H3_MIN_SECONDS = 4
H3_MAX_SECONDS = 15
H3_DEFAULT_STEPS = 50
H3_DEFAULT_VIDEO_SHIFT = 12.0  # model_index.json 的 sigma_shift_scales.video
H3_DEFAULT_AUDIO_SHIFT = 3.0  # model_index.json 的 sigma_shift_scales.audio
H3_DEFAULT_SEED = 42
H3_ASPECT_RATIOS: dict[str, float] = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}
# Ref2VA 参考视频约束（reference_video.py::_validate_reference_video_metadata）：
# 时长 2~15 秒、fps 23.976~60、短边 >=256 且长边 <=5760、宽高比 0.4~2.5、
# 容器 MP4/MOV、视频编码 H.264/H.265、有音轨时必须 AAC/MP3、体积 <=50MiB。
# 导演模式切出的"上一镜头尾部"必须同时满足这些，尤其是 2 秒下限。
REF_VIDEO_MIN_SECONDS = 2.0
REF_VIDEO_MAX_SECONDS = 15.0

# 参考图约束：_validate_reference_image / _reference_image_shape
H3_IMAGE_MAX_BYTES = 30 * 1024 * 1024
H3_IMAGE_MIN_EDGE = 256
H3_IMAGE_MAX_EDGE = 5760
H3_IMAGE_MIN_RATIO = 0.4
H3_IMAGE_MAX_RATIO = 2.5
H3_IMAGE_FORMATS = {"jpeg", "png", "webp", "heic", "heif"}

# FL2VA 的三种模式由 extra_args.frame_indices 决定，pipeline 只接受这三种组合
# （_resolve_fl2va_keyframe_indices）。单图默认是 [0]，所以「尾帧」模式必须显式
# 传 [-1] 才拿得到——不传就静默变成首帧模式。
H3_FL2VA_MODES: dict[str, dict[str, Any]] = {
    "first": {
        "label": "首帧生视频",
        "hint": "上传一张图作为第一帧，向后演绎",
        "frame_indices": [0],
        "labels": ["首帧"],
    },
    "last": {
        "label": "尾帧生视频",
        "hint": "上传一张图作为最后一帧，倒推出前面的过程",
        "frame_indices": [-1],
        "labels": ["尾帧"],
    },
    "first_last": {
        "label": "首尾帧生视频",
        "hint": "上传两张图，在首尾之间补全运动",
        "frame_indices": [0, -1],
        "labels": ["首帧", "尾帧"],
    },
}

# ETA 估算：t = k * work^alpha，work = 帧数 x 像素 x 步数。
# 缺省值由两个历史样本拟合得出（459s @ 5s/1024x576、787s @ 4s/1344x768），
# 每完成一个任务就用真实 inference_time_s 重新拟合，样本越多越准。
ETA_FALLBACK_ALPHA = 1.6
ETA_FALLBACK_K = 787.3 / (192 * 1024 * 768 * H3_DEFAULT_STEPS) ** ETA_FALLBACK_ALPHA
ETA_ALPHA_BOUNDS = (1.0, 2.2)

# 真实步进来自后端日志里的 tqdm 行：" 24%|##4       | 12/50 [00:00<00:00, ...it/s]"
# H3 的去噪循环用 progress_bar(total=去噪步数)，只写 rank0 的 stderr，不进 API。
# run_h3.sh 把 stderr 落到 BACKEND_LOG 后，这里从尾部反向找最后一条。
TQDM_STEP_PATTERN = re.compile(
    r"(\d+)/(\d+)\s*\[((?:\d+:)?\d+:\d+)<[^\]]*(?:it/s|s/it)"
)
LOG_TAIL_BYTES = 262144

jobs: dict[str, dict[str, Any]] = {}
watchers: dict[str, asyncio.Task[None]] = {}
decode_recoveries: set[str] = set()
jobs_lock = asyncio.Lock()
director_projects: dict[str, dict[str, Any]] = {}
director_lock = asyncio.Lock()
director_scheduler_task: asyncio.Task[None] | None = None
library_items: dict[str, dict[str, Any]] = {}
library_lock = asyncio.Lock()
model_switch_task: asyncio.Task[None] | None = None
model_switch_lock = asyncio.Lock()
model_state_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# H3 形状计算 —— 复刻后端算法，用于提交前预测输出尺寸与时长
# ---------------------------------------------------------------------------
def align_frames(frame_count: int) -> int:
    """向上取整到 H3 的 17n+5 帧边界（time_request.py::_align_frame_count）。"""
    if frame_count <= 0:
        return 1
    aligned = int(frame_count)
    while aligned % H3_FRAME_MODULUS != H3_FRAME_REMAINDER:
        aligned += 1
    return aligned


def _align_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def h3_canvas(ratio: float) -> tuple[int, int]:
    """按官方配方把宽高比解析成画布，返回 (width, height)。

    复刻 pipeline_minimax_h3.py::_resolve_output_canvas，外加 _resolve_shape
    随后做的一次 //32*32 截断。
    """
    if ratio >= 1.0:
        width, height = H3_SHORT_EDGE * ratio, float(H3_SHORT_EDGE)
    else:
        width, height = float(H3_SHORT_EDGE), H3_SHORT_EDGE / ratio
    area = width * height
    if area > H3_MAX_PIXELS:
        scale = (H3_MAX_PIXELS / area) ** 0.5
        width *= scale
        height *= scale
    return _align_multiple(width) // 32 * 32, _align_multiple(height) // 32 * 32


def canvas_for_ratio_name(name: str) -> tuple[int, int]:
    return h3_canvas(H3_ASPECT_RATIOS[name])


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    task: Literal["t2va", "fl2va"] = "t2va"
    # FL2VA 的关键帧编排；task=t2va 时忽略
    fl2va_mode: Literal["first", "last", "first_last"] = "first"
    # t2va 必须显式给出宽高比，否则 pipeline 直接报
    # "t2va requires an explicit aspect_ratio"；fl2va 由首帧图片推导，此值被忽略。
    aspect_ratio: Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = "16:9"
    seconds: int = Field(default=8, ge=H3_MIN_SECONDS, le=H3_MAX_SECONDS)
    quality: Literal["lossless", "high"] = "lossless"
    seed: int | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    generate_sound: bool = True
    sound_duration: float | None = Field(default=None, gt=0)
    # 仅在 quality=high（Cache-DiT 生效）时可用，否则 pipeline 会报
    # "force-refresh arguments require an active Cache-DiT request target"
    force_refresh_step_hint: int | None = Field(default=None, ge=1)
    force_refresh_step_policy: Literal["once", "repeat"] | None = None
    enable_frame_interpolation: bool = False
    frame_interpolation_exp: int | None = Field(default=None, ge=1, le=4)
    frame_interpolation_scale: float | None = Field(default=None, gt=0, le=2)
    # Internal recovery control; never persisted into the reusable editor
    # snapshot. The worker validates the checkpoint against prompt and shape.
    resume_checkpoint_key: str | None = Field(default=None, max_length=180, exclude=True)

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("提示词不能为空")
        return value

    @property
    def steps(self) -> int:
        return self.num_inference_steps or H3_DEFAULT_STEPS

    @property
    def frames(self) -> int:
        return align_frames(self.seconds * H3_FPS)

    @property
    def frame_labels(self) -> list[str]:
        return list(H3_FL2VA_MODES[self.fl2va_mode]["labels"])

    def validate_contract(self, image_count: int) -> None:
        """把后端会在 GPU 上跑几秒后才报的错误提前到提交时报出来。"""
        if self.task == "t2va":
            if image_count:
                raise HTTPException(status_code=400, detail="文生视频模式不接受参考图片，请切换到图生视频")
        else:
            mode = H3_FL2VA_MODES[self.fl2va_mode]
            expected = len(mode["labels"])
            if image_count != expected:
                labels = "、".join(self.frame_labels)
                raise HTTPException(
                    status_code=400,
                    detail=f"{mode['label']}需要正好 {expected} 张图片（{labels}），当前 {image_count} 张",
                )
        if self.force_refresh_step_hint is not None:
            if self.quality != "high":
                raise HTTPException(
                    status_code=400,
                    detail="Cache-DiT 刷新提示只在 quality=high 时可用，请先切换到 High 档",
                )
            if self.force_refresh_step_hint > self.steps:
                raise HTTPException(
                    status_code=400,
                    detail=f"刷新步提示必须在 1 到步数（{self.steps}）之间",
                )
        if self.force_refresh_step_policy is not None and self.force_refresh_step_hint is None:
            raise HTTPException(status_code=400, detail="刷新策略需要同时给出刷新步提示")
        if self.sound_duration is not None and not self.generate_sound:
            raise HTTPException(status_code=400, detail="音频时长只在开启生成音频时有意义")


class MoveJobRequest(BaseModel):
    direction: Literal["up", "down"]


class ModelSwitchRequest(BaseModel):
    partition: Literal["ref2va", "fl2va"]


class ReferenceMeta(BaseModel):
    client_id: str = Field(default="", max_length=80)
    kind: Literal["image", "video", "audio"]
    label: str = Field(min_length=1, max_length=80)
    role: str = Field(min_length=1, max_length=60)
    subject: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=1000)
    priority: Literal["primary", "support", "background"] = "support"
    enabled: bool = True


class Ref2VARequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    aspect_ratio: Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = "16:9"
    seconds: int = Field(default=5, ge=H3_MIN_SECONDS, le=H3_MAX_SECONDS)
    quality: Literal["lossless", "high"] = "lossless"
    seed: int | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    generate_sound: bool = True
    sound_duration: float | None = Field(default=None, gt=0)
    start_time_seconds: float | None = Field(default=None, ge=0)
    references_meta: list["ReferenceMeta"] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def strip_ref_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("提示词不能为空")
        return value

    @property
    def steps(self) -> int:
        return self.num_inference_steps or H3_DEFAULT_STEPS

    @property
    def frames(self) -> int:
        return align_frames(self.seconds * H3_FPS)


class DirectorProjectRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    synopsis: str = Field(default="", max_length=20000)
    visual_bible: str = Field(default="", max_length=20000)
    identity_bible: str = Field(default="", max_length=20000)
    style_bible: str = Field(default="", max_length=20000)
    shot_plan: str = Field(default="", max_length=50000)
    aspect_ratio: Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = "16:9"
    default_seconds: int = Field(default=5, ge=H3_MIN_SECONDS, le=H3_MAX_SECONDS)
    quality: Literal["lossless", "high"] = "lossless"
    seed: int = 42
    # 这个值会被切成"上一镜头尾部"喂给 Ref2VA 当参考视频，而参考视频时长
    # 硬下限是 2 秒（reference_video.py::_validate_reference_video_metadata）。
    # 低于 2 秒的话第二个镜头必定报 "reference video 0 duration must be in [2, 15]"。
    overlap_seconds: float = Field(default=2.0, ge=2.0, le=4.0)
    # H3 原生声音无法稳定按角色区分男女声。导演台默认关闭，避免把
    # 不可控的单一声线误当成角色对白；需要时仍可显式选择 native。
    generate_sound: bool = False
    sound_mode: Literal["off", "native"] = "off"
    review_mode: Literal["pipeline", "review_gate"] = "pipeline"
    # Legacy field kept for saved clients; review_mode is authoritative.
    auto_approve: bool = False


class DirectorProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    synopsis: str | None = Field(default=None, max_length=20000)
    visual_bible: str | None = Field(default=None, max_length=30000)
    identity_bible: str | None = Field(default=None, max_length=30000)
    style_bible: str | None = Field(default=None, max_length=30000)
    aspect_ratio: Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] | None = None
    default_seconds: int | None = Field(default=None, ge=H3_MIN_SECONDS, le=H3_MAX_SECONDS)
    quality: Literal["lossless", "high"] | None = None
    seed: int | None = Field(default=None, ge=0)
    overlap_seconds: float | None = Field(default=None, ge=2.0, le=4.0)
    generate_sound: bool | None = None
    sound_mode: Literal["off", "native"] | None = None
    review_mode: Literal["pipeline", "review_gate"] | None = None
    auto_approve: bool | None = None


class DirectorShotRequest(BaseModel):
    title: str = Field(default="新镜头", max_length=120)
    prompt: str = Field(min_length=1, max_length=20000)
    seconds: int = Field(default=5, ge=H3_MIN_SECONDS, le=H3_MAX_SECONDS)
    seed_offset: int = Field(default=0, ge=0, le=1000000)
    continuity: Literal["auto", "previous_video", "character_images", "independent"] = "auto"
    character_ids: list[str] = Field(default_factory=list)
    location_id: str | None = None
    scene_asset_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    start_state: str = Field(default="", max_length=2000)
    end_state: str = Field(default="", max_length=2000)
    camera: str = Field(default="", max_length=2000)
    sound: str = Field(default="", max_length=2000)
    trim_start: float = Field(default=0.0, ge=0, le=60)
    trim_end: float | None = Field(default=None, gt=0, le=60)
    audio_gain: float = Field(default=1.0, ge=0, le=3)


class DirectorEntityRequest(BaseModel):
    kind: Literal["character", "location", "style", "audio"]
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=5000)
    locked_traits: str = Field(default="", max_length=5000)
    asset_ids: list[str] = Field(default_factory=list)


class DirectorAssetUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: str = Field(default="identity", max_length=60)
    subject: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=2000)
    priority: Literal["primary", "support", "background"] = "support"
    enabled: bool = True


class LibraryItemUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    role: str = Field(default="", max_length=60)
    subject: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=2000)
    priority: Literal["primary", "support", "background"] = "support"
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip()[:24] for item in value if item.strip()]
        return cleaned[:12]


class LibraryImportRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1, max_length=12)


class DirectorReviewRequest(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=2000)


class DirectorAssembleRequest(BaseModel):
    transition: Literal["cut", "crossfade"] = "cut"


class DirectorShotEditRequest(BaseModel):
    """Non-destructive edit settings applied only when the project is assembled."""

    trim_start: float = Field(default=0.0, ge=0, le=60)
    trim_end: float | None = Field(default=None, gt=0, le=60)
    audio_gain: float = Field(default=1.0, ge=0, le=3)


class DirectorMoveRequest(BaseModel):
    direction: Literal["up", "down"]


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def _api_key() -> str:
    try:
        return H3_PROVIDER.api_key()
    except OSError as exc:
        raise RuntimeError(f"无法读取 H3 API key: {exc}") from exc


def _headers() -> dict[str, str]:
    return H3_PROVIDER.headers()


def _load_jobs() -> None:
    if not JOBS_FILE.exists():
        return
    try:
        data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            jobs.update(data)
    except (OSError, json.JSONDecodeError):
        pass


def _model_state_default() -> dict[str, Any]:
    return {
        "partition": H3_PARTITION,
        "target": H3_PARTITION,
        "status": "ready",
        "stage": "ready",
        "progress": 100,
        "message": f"{H3_PARTITION.upper()} 模型已就绪",
        "error": None,
        "started_at": None,
        "updated_at": int(time.time()),
    }


def _load_model_state() -> dict[str, Any]:
    try:
        payload = json.loads(MODEL_STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            state = _model_state_default()
            state.update(payload)
            return state
    except (OSError, json.JSONDecodeError):
        pass
    return _model_state_default()


model_state: dict[str, Any] = _load_model_state()


def _save_model_state_unlocked() -> None:
    temporary = MODEL_STATE_FILE.with_name(f".{MODEL_STATE_FILE.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(model_state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MODEL_STATE_FILE)


def _sync_model_state_from_disk() -> None:
    try:
        payload = json.loads(MODEL_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(payload, dict):
        model_state.update(payload)


async def _set_model_state(**changes: Any) -> None:
    async with model_state_lock:
        model_state.update(changes)
        model_state["updated_at"] = int(time.time())
        _save_model_state_unlocked()


def _model_switch_command(partition: str) -> list[str]:
    # 由当前前端 API 进程启动一个脱离父进程的脚本；脚本负责优雅终止旧
    # h3-api、更新状态文件、启动新分区并按阶段轮询 8000/health。
    return ["/bin/bash", str(ROOT / "switch_h3_model.sh"), partition]


def _load_library() -> None:
    """启动时载入素材库索引，顺手清掉文件已丢失的条目。"""
    if not LIBRARY_FILE.is_file():
        return
    try:
        data = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    for item_id, item in data.items():
        if not isinstance(item, dict):
            continue
        path = LIBRARY_ROOT / str(item.get("filename") or "")
        if not path.is_file():
            # 文件被手工删掉的话索引留着只会让前端渲染出打不开的卡片
            continue
        library_items[str(item_id)] = item


def _save_library_unlocked() -> None:
    tmp = LIBRARY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(library_items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LIBRARY_FILE)


def _library_path(item: dict[str, Any]) -> Path:
    """解析素材文件路径，并挡住通过 filename 越出库目录的尝试。"""
    root = LIBRARY_ROOT.resolve()
    path = (root / str(item.get("filename") or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="无效素材路径") from exc
    return path


def _library_view(item: dict[str, Any]) -> dict[str, Any]:
    view = dict(item)
    view["url"] = f"/api/library/items/{item['id']}/file"
    return view


def _output_path(filename: str) -> Path:
    """Resolve a generated filename in the organized output directory.

    The root-level fallback keeps old job records usable before migration.
    """
    name = Path(str(filename)).name
    organized = OUTPUT_ROOT / name
    legacy = ROOT / name
    return legacy if legacy.exists() and not organized.exists() else organized


def _safe_output_video_path(filename: str) -> Path:
    """Resolve one generated MP4 without allowing paths outside the output roots."""
    raw_name = str(filename)
    safe_name = Path(raw_name).name
    if safe_name != raw_name or not safe_name.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="无效视频文件名")
    return _output_path(safe_name)


def _load_director_projects() -> None:
    if not DIRECTOR_FILE.exists():
        return
    try:
        data = json.loads(DIRECTOR_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for project in data.values():
                project.setdefault("entities", [])
                project.setdefault("auto_approve", False)
                project.setdefault("sound_mode", "native" if project.get("generate_sound", True) else "off")
                project.setdefault("identity_bible", "")
                project.setdefault("style_bible", "")
                # New projects default to non-blocking pipeline generation.
                # Older records without this field retain their data and gain
                # the new default; review_gate can be selected explicitly.
                project.setdefault("review_mode", "pipeline")
                for asset in project.get("assets", []):
                    asset.setdefault("role", "identity" if asset.get("kind") == "image" else "motion")
                    asset.setdefault("subject", "")
                    asset.setdefault("description", "")
                    asset.setdefault("priority", "support")
                    asset.setdefault("enabled", True)
                for shot in project.get("shots", []):
                    for key, default in {
                        "character_ids": [], "location_id": None, "asset_ids": [],
                        "scene_asset_id": None,
                        "start_state": "", "end_state": "", "camera": "", "sound": "",
                        "review_note": "",
                        "execution": None,
                    }.items():
                        shot.setdefault(key, default)
            director_projects.update(data)
    except (OSError, json.JSONDecodeError):
        pass


def _project_review_mode(project: dict[str, Any]) -> str:
    mode = str(project.get("review_mode") or "pipeline")
    return mode if mode in {"pipeline", "review_gate"} else "pipeline"


def _director_sound_mode(project: dict[str, Any]) -> str:
    """Return the explicit director audio contract.

    ``generate_sound`` is retained for old project files, but new execution
    snapshots use the named mode so the UI and backend cannot silently disagree.
    """
    mode = str(project.get("sound_mode") or "").lower()
    if mode in {"off", "native"}:
        return mode
    return "native" if project.get("generate_sound", True) else "off"


def _director_sound_enabled(project: dict[str, Any]) -> bool:
    return _director_sound_mode(project) == "native"


def _director_shot_prompt(project: dict[str, Any], shot: dict[str, Any], index: int) -> str:
    """Build the compact, time-budgeted prompt actually sent to H3."""
    seconds = int(shot.get("seconds") or 5)
    action = str(shot.get("prompt") or "").strip()
    entities = {item["id"]: item for item in project.get("entities", [])}
    selected_ids = list(shot.get("character_ids", []))
    locks = []
    for item_id in selected_ids:
        entity = entities.get(item_id)
        if entity:
            locks.append(f"{entity['kind']} {entity['name']}：{entity.get('locked_traits') or entity.get('description') or ''}")
    scene_asset = next(
        (item for item in project.get("assets", []) if item.get("id") == shot.get("scene_asset_id")),
        None,
    )
    inherits_previous = index > 0 and shot.get("continuity") in {"auto", "previous_video"}
    if scene_asset:
        current_scene = (
            f"当前场景严格以绑定的场景参考图“{scene_asset.get('name')}”为唯一背景依据；"
            "不得改造、切换或展示其它场景。"
        )
    elif inherits_previous:
        current_scene = (
            "本镜头没有指定新场景图：完整继承上一镜头最后画面中的场景、光线、空间结构与人物站位；"
            "不得自行切换地点或创造新背景。"
        )
    else:
        current_scene = "本镜头没有独立场景图，只使用已绑定身份素材中的环境，不得自行切换多个地点。"
    parts = [
        f"导演短镜头{index + 1}，时长约{seconds}秒。",
        "固定一个场景，不切场、不转场、不蒙太奇、不瞬移；只完成一个主要动作，结尾停在明确姿态，动作速度自然。",
        f"全片身份锁定：{str(project.get('identity_bible') or '').strip()}",
        f"全片视觉风格：{str(project.get('style_bible') or '').strip()}",
        "；".join(locks),
        current_scene,
        f"开始状态：{str(shot.get('start_state') or '按上一镜头尾帧自然承接')}",
        f"主要动作：{action}",
        f"结束状态：{str(shot.get('end_state') or '保持最后姿态供下一镜承接')}",
        f"镜头：{str(shot.get('camera') or '稳定中景，轻微跟随，不快速切换')}",
    ]
    return "\n".join(part for part in parts if part)


def _save_director_projects_unlocked() -> None:
    temporary = DIRECTOR_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(director_projects, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(DIRECTOR_FILE)


def _director_asset_path(project_id: str, relative: str) -> Path:
    project_root = (DIRECTOR_ASSET_ROOT / project_id).resolve()
    path = (project_root / relative).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="无效导演素材路径") from exc
    return path


def _canvas_pixels(ratio_name: str | None) -> int:
    """按宽高比名算出画布像素数，用于预测尚未提交镜头的耗时。"""
    if ratio_name in H3_ASPECT_RATIOS:
        width, height = canvas_for_ratio_name(ratio_name)
        return width * height
    return 1344 * 768


def _director_project_view(project: dict[str, Any]) -> dict[str, Any]:
    view = json.loads(json.dumps(project, ensure_ascii=False))
    view["review_mode"] = _project_review_mode(project)
    shots = view.get("shots") or []
    project_id = str(view.get("id") or "")
    view["duration_seconds"] = sum(float(shot.get("seconds") or 0) for shot in shots)
    for shot in shots:
        shot.setdefault("trim_start", 0.0)
        shot.setdefault("trim_end", None)
        shot.setdefault("audio_gain", 1.0)
        shot["video_url"] = (
            f"/api/director/projects/{project_id}/shots/{shot['id']}/video"
            if shot.get("filename") else None
        )
        try:
            shot["model_input_preview"] = _director_shot_prompt(
                project, shot, int(shot.get("order") or 0)
            )
        except Exception as exc:
            shot["model_input_preview"] = f"预览生成失败：{exc}"
    view["completed_shots"] = sum(shot.get("status") in {"approved", "completed"} for shot in shots)
    view["total_shots"] = len(shots)
    # 把对应任务的实时进度挂到镜头上：导演台否则只能显示 "in_progress"，
    # 用户得切回队列页才知道跑到第几步。_decorate_job 已经算好了真实步数与 ETA。
    model = _eta_model()
    remaining = 0.0
    for shot in shots:
        record = jobs.get(str(shot.get("job_id") or ""))
        if record is None:
            continue
        decorated = _decorate_job(record, model)
        shot["progress_display"] = decorated.get("progress_display")
        shot["step"] = decorated.get("step")
        shot["total_steps"] = decorated.get("total_steps")
        shot["eta_remaining_s"] = decorated.get("eta_remaining_s")
        if decorated.get("eta_remaining_s"):
            remaining += float(decorated["eta_remaining_s"])
    # 还没提交的镜头按预测值累加，给出整片剩余时间的粗估。
    for shot in shots:
        if shot.get("status") in {"draft", "ready"}:
            predicted = _predict_seconds(
                {
                    "frames": align_frames(int(shot.get("seconds") or 5) * H3_FPS),
                    "pixels": _canvas_pixels(view.get("aspect_ratio")),
                    "steps": H3_DEFAULT_STEPS,
                },
                model,
            )
            if predicted:
                remaining += predicted
    view["eta_remaining_s"] = round(remaining) if remaining else None
    view["output_url"] = (
        f"/api/director/projects/{project_id}/output"
        if view.get("output_filename") else None
    )
    return view


def _save_jobs_unlocked() -> None:
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(JOBS_FILE)


async def _update_job(job_id: str, **changes: Any) -> None:
    async with jobs_lock:
        record = jobs.setdefault(job_id, {"id": job_id})
        record.update(changes)
        record["updated_at"] = int(time.time())
        _save_jobs_unlocked()


def _error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error") or payload.get("message")
        if isinstance(detail, dict):
            detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
        if detail:
            return detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    return response.text.strip() or f"HTTP {response.status_code}"


# ---------------------------------------------------------------------------
# ETA 估算
#
# 后端的 progress 字段是死的：api_server.py 只在完成时写 100，中途恒为 0，
# denoise_loop 里没有任何回调或日志，/metrics 也没有步进计数器。所以真实百分比
# 拿不到，只能用历史耗时拟合。每个完成的任务都会自动成为新样本。
# ---------------------------------------------------------------------------
def _read_backend_step() -> dict[str, Any] | None:
    """从后端日志尾部取最后一条 tqdm 步进，拿不到就返回 None。

    只读尾部若干字节，日志再长也不会拖慢轮询。日志由 run_h3.sh 每次启动时
    切分，所以不会读到上一个后端进程的残留步数。
    """
    try:
        size = BACKEND_LOG.stat().st_size
        with BACKEND_LOG.open("rb") as handle:
            if size > LOG_TAIL_BYTES:
                handle.seek(size - LOG_TAIL_BYTES)
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    matches = TQDM_STEP_PATTERN.findall(tail)
    if not matches:
        return None
    current_text, total_text, elapsed_text = matches[-1]
    current, total = int(current_text), int(total_text)
    if total <= 0 or current > total:
        return None
    elapsed_parts = [int(part) for part in elapsed_text.split(":")]
    elapsed_seconds = 0
    for part in elapsed_parts:
        elapsed_seconds = elapsed_seconds * 60 + part
    return {
        "step": current,
        "total_steps": total,
        "denoise_elapsed_s": elapsed_seconds,
        "log_mtime": int(BACKEND_LOG.stat().st_mtime),
    }


def _job_work(record: dict[str, Any]) -> float | None:
    """算出一个任务的计算量代理值：帧数 x 像素 x 步数。"""
    frames = record.get("frames")
    pixels = record.get("pixels")
    if not frames or not pixels:
        # 兼容新增字段之前写下的旧记录
        seconds = record.get("seconds")
        size = record.get("size") or record.get("expected_size")
        if not seconds or not isinstance(size, str) or "x" not in size:
            return None
        try:
            width, height = (int(part) for part in size.split("x", 1))
        except ValueError:
            return None
        frames = align_frames(int(seconds) * H3_FPS)
        pixels = width * height
    steps = record.get("steps") or H3_DEFAULT_STEPS
    work = float(frames) * float(pixels) * float(steps)
    return work if work > 0 else None


def _eta_model() -> dict[str, Any]:
    """用所有已完成任务拟合 log t = log k + alpha * log work。"""
    samples: list[tuple[float, float, str]] = []
    for record in jobs.values():
        elapsed = record.get("inference_time_s")
        if record.get("status") != "completed" or not elapsed or elapsed <= 0:
            continue
        work = _job_work(record)
        if work is None:
            continue
        samples.append((work, float(elapsed), str(record.get("quality") or "lossless")))

    if not samples:
        return {"k": ETA_FALLBACK_K, "alpha": ETA_FALLBACK_ALPHA, "samples": 0, "calibrated": False}

    alpha = ETA_FALLBACK_ALPHA
    if len(samples) >= 2:
        xs = [math.log(work) for work, _, _ in samples]
        ys = [math.log(elapsed) for _, elapsed, _ in samples]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        variance = sum((x - mean_x) ** 2 for x in xs)
        if variance > 1e-9:
            covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
            alpha = min(max(covariance / variance, ETA_ALPHA_BOUNDS[0]), ETA_ALPHA_BOUNDS[1])

    # k 取几何均值，避免单个异常样本把整条曲线拽偏
    def geometric_k(subset: list[tuple[float, float, str]]) -> float:
        logs = [math.log(elapsed) - alpha * math.log(work) for work, elapsed, _ in subset]
        return math.exp(sum(logs) / len(logs))

    k_by_quality = {
        quality: geometric_k(subset)
        for quality in {sample[2] for sample in samples}
        if (subset := [s for s in samples if s[2] == quality])
    }
    return {
        "k": geometric_k(samples),
        "alpha": alpha,
        "k_by_quality": k_by_quality,
        "samples": len(samples),
        "calibrated": True,
    }


def _predict_seconds(record: dict[str, Any], model: dict[str, Any]) -> float | None:
    work = _job_work(record)
    if work is None:
        return None
    quality = str(record.get("quality") or "lossless")
    k = model.get("k_by_quality", {}).get(quality) or model["k"]
    return k * work ** model["alpha"]


def _execution_owner_id() -> str | None:
    """Return the only job that can own the single H3 denoise loop.

    vLLM-Omni reports both the running request and FIFO waiters as
    in_progress. The backend tqdm log is global, so only the oldest active
    request may receive its step counter.
    """
    candidates = [
        record
        for record in jobs.values()
        if record.get("status") not in TERMINAL_STATUSES
    ]
    if not candidates:
        return None
    # dict preserves submission order; min is stable when several requests
    # share the same integer-second created_at value.
    owner = min(candidates, key=lambda item: item.get("created_at", 0))
    return str(owner.get("id")) if owner.get("id") else None


def _decorate_job(record: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """给任务记录挂上进度视图，不改动持久化的原始字段。"""
    view = dict(record)
    checkpoint_dir = LATENT_CHECKPOINT_ROOT / str(record.get("id") or "")
    view["decode_checkpoint_available"] = all(
        (checkpoint_dir / f"rank{rank}.pt").is_file() for rank in (0, 1)
    )
    status = str(record.get("status") or "queued")
    predicted = _predict_seconds(record, model)
    view["eta_predicted_s"] = predicted
    view["eta_samples"] = model["samples"]
    view["eta_calibrated"] = model["calibrated"]

    if status == "completed":
        view["progress_display"] = 100
        view["progress_source"] = "done"
        return view
    if status in TERMINAL_STATUSES:
        view["progress_display"] = 0
        view["progress_source"] = "none"
        return view

    execution_owner = _execution_owner_id()
    if execution_owner and str(record.get("id")) != execution_owner:
        view["status"] = "queued"
        view["execution_status"] = "queued"
        view["progress_display"] = 0
        view["progress_source"] = "queue"
        return view
    view["execution_status"] = "in_progress"

    started = record.get("started_at") or record.get("created_at")
    if not started:
        view["progress_display"] = 0
        view["progress_source"] = "none"
        return view
    elapsed = max(0.0, time.time() - float(started))
    view["elapsed_s"] = elapsed

    # 真实步进优先。日志里的 tqdm 是全局的（后端一次只跑一个任务），只有当它
    # 比任务开始时间新时才可信 —— 否则那是上一个任务留下的行。
    step = _read_backend_step() if status == "in_progress" else None
    if step and step["log_mtime"] >= int(float(started)):
        view["step"] = step["step"]
        view["total_steps"] = step["total_steps"]
        view["denoise_elapsed_s"] = step["denoise_elapsed_s"]
        view["progress_display"] = min(99, int(step["step"] / step["total_steps"] * 100))
        view["progress_source"] = "steps"
        if step["step"] > 0:
            per_step = elapsed / step["step"]
            view["eta_remaining_s"] = max(0.0, per_step * (step["total_steps"] - step["step"]))
        return view

    if predicted and predicted > 0:
        # 封顶 99%：真实完成信号只来自后端，估算值不允许宣布完成
        view["progress_display"] = min(99, int(elapsed / predicted * 100))
        view["eta_remaining_s"] = max(0.0, predicted - elapsed)
        view["progress_source"] = "eta"
    else:
        view["progress_display"] = 0
        view["progress_source"] = "none"
    return view


# ---------------------------------------------------------------------------
# 任务轮询
# ---------------------------------------------------------------------------
async def _download_video(client: httpx.AsyncClient, job_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"h3_{stamp}_{job_id[-8:]}.mp4"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / filename
    partial = OUTPUT_ROOT / f".{filename}.part"

    async with client.stream(
        "GET",
        f"{H3_BASE_URL}/v1/videos/{job_id}/content",
        headers=_headers(),
        timeout=httpx.Timeout(120.0, read=None),
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise RuntimeError(body.decode("utf-8", errors="replace") or f"下载失败 HTTP {response.status_code}")
        with partial.open("wb") as output:
            async for chunk in response.aiter_bytes(1024 * 1024):
                output.write(chunk)
    partial.replace(destination)
    return filename


async def _recover_stuck_decode(old_job_id: str) -> None:
    if old_job_id in decode_recoveries:
        return
    decode_recoveries.add(old_job_id)
    record = jobs.get(old_job_id)
    if record is None:
        return
    snapshot = _request_snapshot(record)
    snapshot["resume_checkpoint_key"] = old_job_id
    assets = _saved_assets(record)
    if len(assets) != int(record.get("reference_images") or 0):
        await _update_job(old_job_id, status="failed", message="checkpoint 恢复所需参考素材不完整")
        return
    multipart: list[tuple[str, Any]] = [("payload", (None, json.dumps(snapshot, ensure_ascii=False)))]
    for path in assets:
        multipart.append(("images", (path.name, path.read_bytes(), "application/octet-stream")))
    try:
        # The request-mode worker is still occupied waiting to return the old
        # output. Submitting a decode-only request to that same process merely
        # queues behind the stuck RPC. Restart the local partition first; the
        # latent checkpoint is on disk and survives the process replacement.
        await _set_model_state(
            partition=H3_PARTITION, target=H3_PARTITION, status="loading",
            stage="stopping", progress=3,
            message="输出回传超时，正在重启引擎并从 checkpoint 恢复解码…",
            error=None, started_at=int(time.time()),
        )
        subprocess.Popen(
            _model_switch_command(H3_PARTITION), cwd=str(ROOT),
            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        await asyncio.sleep(8)
        deadline = time.monotonic() + 1800
        async with httpx.AsyncClient(timeout=10.0) as health_client:
            while time.monotonic() < deadline:
                try:
                    health = await health_client.get(f"{H3_BASE_URL}/health", headers=_headers())
                    if health.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(5)
            else:
                raise RuntimeError("重启 H3 后 30 分钟内未恢复就绪")
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{FRONTEND_INTERNAL_URL}/api/generate", files=multipart)
        if response.status_code != 200:
            raise RuntimeError(_error_text(response))
        new_id = str(response.json()["id"])
        await _update_job(old_job_id, status="cancelled", message="输出阶段超时，已转入 checkpoint 解码恢复")
        for project in director_projects.values():
            for shot in project.get("shots", []):
                if str(shot.get("job_id") or "") == old_job_id:
                    shot["job_id"] = new_id
                    shot["status"] = "queued"
                    shot["message"] = "已从去噪 checkpoint 恢复解码"
                    project["updated_at"] = int(time.time())
        _start_watcher(new_id)
        async with director_lock:
            _save_director_projects_unlocked()
    except Exception as exc:
        await _update_job(old_job_id, status="failed", message=f"checkpoint 恢复失败：{exc}")


async def _watch_job(job_id: str) -> None:
    try:
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                try:
                    response = await client.get(f"{H3_BASE_URL}/v1/videos/{job_id}", headers=_headers())
                    if response.status_code == 404:
                        # 任务已被后端删除（多半是从别处 DELETE 掉了）
                        await _update_job(job_id, status="cancelled", message="任务已从 H3 后端移除")
                        return
                    if response.status_code == 503:
                        raise RuntimeError(_error_text(response))
                    # 失败的任务以非 200 状态码返回，但 body 仍是完整的任务记录
                    # （api_server.py::retrieve_video 按 error.code 决定状态码）
                    payload = response.json()
                    if not isinstance(payload, dict) or "status" not in payload:
                        raise RuntimeError(_error_text(response))

                    status = str(payload.get("status", "running")).lower()
                    changes: dict[str, Any] = {
                        "status": status,
                        "inference_time_s": payload.get("inference_time_s"),
                        "peak_memory_mb": payload.get("peak_memory_mb"),
                        "stage_durations": payload.get("stage_durations") or None,
                        "message": None,
                    }
                    if status == "in_progress" and not jobs.get(job_id, {}).get("started_at"):
                        changes["started_at"] = int(time.time())
                    error = payload.get("error")
                    if error:
                        changes["message"] = error.get("message") if isinstance(error, dict) else str(error)
                    await _update_job(job_id, **changes)

                    # A local H3 request can finish denoise and expose its
                    # post-denoise checkpoint while the backend's final
                    # output IPC is still stuck.  Do not leave the director
                    # permanently in_progress: after a bounded grace period
                    # recover the checkpoint through the normal decode path.
                    if (
                        status == "in_progress"
                        and H3_PROVIDER.is_local
                        and (LATENT_CHECKPOINT_ROOT / job_id / "rank0.pt").is_file()
                        and (LATENT_CHECKPOINT_ROOT / job_id / "rank1.pt").is_file()
                    ):
                        checkpoint_files = [
                            LATENT_CHECKPOINT_ROOT / job_id / "rank0.pt",
                            LATENT_CHECKPOINT_ROOT / job_id / "rank1.pt",
                        ]
                        # Decode itself can take several minutes. Measure the
                        # grace period from the newest checkpoint write, not
                        # from job start (which includes the whole denoise).
                        checkpoint_age = time.time() - max(path.stat().st_mtime for path in checkpoint_files)
                        decode_stuck_timeout = max(
                            7200,
                            int(config_section("local").get("decode_stuck_timeout_seconds", 7200)),
                        )
                        if checkpoint_age > decode_stuck_timeout:
                            await _recover_stuck_decode(job_id)
                            return

                    if status == "completed":
                        filename = await _download_video(client, job_id)
                        await _update_job(job_id, filename=filename, status="completed")
                        shutil.rmtree(LATENT_CHECKPOINT_ROOT / job_id, ignore_errors=True)
                        return
                    if status in TERMINAL_STATUSES:
                        return
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    await _update_job(job_id, status="waiting_backend", message=str(exc))
                await asyncio.sleep(4)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _update_job(job_id, status="failed", message=str(exc))
    finally:
        watchers.pop(job_id, None)


def _start_watcher(job_id: str) -> None:
    current = watchers.get(job_id)
    if current is None or current.done():
        watchers[job_id] = asyncio.create_task(_watch_job(job_id))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global director_scheduler_task
    _load_jobs()
    _load_director_projects()
    _load_library()
    for job_id, record in list(jobs.items()):
        if record.get("status") not in TERMINAL_STATUSES:
            _start_watcher(job_id)
    director_scheduler_task = asyncio.create_task(_director_scheduler())
    yield
    if director_scheduler_task is not None:
        director_scheduler_task.cancel()
    for task in list(watchers.values()):
        task.cancel()
    pending = list(watchers.values())
    if director_scheduler_task is not None:
        pending.append(director_scheduler_task)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


app = FastAPI(title="H3 视频生成工作台", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    # 工作台是单文件应用，按钮与路由经常随修复更新；禁止浏览器继续使用旧 HTML/JS。
    return FileResponse(
        INDEX_FILE,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/meta")
async def meta() -> dict[str, Any]:
    """把 H3 契约交给前端，避免两边硬编码不同步。"""
    durations = []
    for seconds in range(H3_MIN_SECONDS, H3_MAX_SECONDS + 1):
        frames = align_frames(seconds * H3_FPS)
        durations.append(
            {
                "seconds": seconds,
                "frames": frames,
                # 帧数被对齐到 17n+5，所以实际时长几乎总是比请求值长一点
                "actual_seconds": round(frames / H3_FPS, 3),
                "exact": frames == seconds * H3_FPS,
            }
        )
    ratios = []
    for name, value in H3_ASPECT_RATIOS.items():
        width, height = canvas_for_ratio_name(name)
        ratios.append({"name": name, "width": width, "height": height, "pixels": width * height, "ratio": value})
    return {
        "fps": H3_FPS,
        "short_edge": H3_SHORT_EDGE,
        "max_pixels": H3_MAX_PIXELS,
        "aspect_ratios": ratios,
        "durations": durations,
        "tasks": [
            {
                "value": "t2va",
                "label": "文生视频",
                "hint": "纯文字驱动，需要选择输出比例",
                "images": 0,
            },
            {
                "value": "fl2va",
                "label": "图生视频",
                "hint": "由参考图驱动，输出比例取自第一张图",
                "images": 2,
            },
        ],
        "fl2va_modes": [
            {
                "value": value,
                "label": mode["label"],
                "hint": mode["hint"],
                "images": len(mode["labels"]),
                "labels": mode["labels"],
                "frame_indices": mode["frame_indices"],
            }
            for value, mode in H3_FL2VA_MODES.items()
        ],
        "defaults": {
            "seconds": 8,
            "steps": H3_DEFAULT_STEPS,
            "seed": H3_DEFAULT_SEED,
            "flow_shift": H3_DEFAULT_VIDEO_SHIFT,
            "audio_flow_shift": H3_DEFAULT_AUDIO_SHIFT,
            "aspect_ratio": "16:9",
            "quality": "lossless",
            "task": "t2va",
            "fl2va_mode": "first",
        },
        "image_limits": {
            "max_bytes": H3_IMAGE_MAX_BYTES,
            "min_edge": H3_IMAGE_MIN_EDGE,
            "max_edge": H3_IMAGE_MAX_EDGE,
            "min_ratio": H3_IMAGE_MIN_RATIO,
            "max_ratio": H3_IMAGE_MAX_RATIO,
            "formats": sorted(H3_IMAGE_FORMATS),
        },
        "eta": _eta_model(),
    }


ENGINE_DEAD_MARKERS = (
    "has no live replica",
    "scheduler is dead",
    "Stage-0 inline diffusion engine is dead",
    "EngineDeadError",
)


def _backend_engine_dead() -> str | None:
    """扫日志尾部判断引擎是否已僵死，返回命中的标记（没死则 None）。

    /health 返回 200 并不代表能干活：TP rank 崩溃后 API 层仍然活着，
    但所有请求都会以 "Stage-0 has no live replica" 失败。这种状态只能靠日志识别。
    run_h3.sh 每次启动都会切分日志，所以不会读到上一个进程周期的残留。
    """
    try:
        size = BACKEND_LOG.stat().st_size
        with BACKEND_LOG.open("rb") as handle:
            if size > LOG_TAIL_BYTES:
                handle.seek(size - LOG_TAIL_BYTES)
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for marker in ENGINE_DEAD_MARKERS:
        if marker in tail:
            return marker
    return None


@app.get("/api/health")
async def health() -> dict[str, Any]:
    _sync_model_state_from_disk()
    running = sum(1 for record in jobs.values() if record.get("status") not in TERMINAL_STATUSES)
    result: dict[str, Any] = {"frontend": "ok", "active_jobs": running}
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(f"{H3_BASE_URL}/health")
            result["backend"] = "ok" if response.status_code == 200 else "unavailable"
            result["backend_status"] = response.status_code
            if response.status_code == 200 and H3_PROVIDER.is_local:
                load = await client.get(f"{H3_BASE_URL}/load", headers=_headers())
                if load.status_code == 200:
                    result["server_load"] = load.json().get("server_load")
                    if model_state.get("status") == "loading" and model_state.get("partition") == H3_PARTITION:
                        await _set_model_state(
                            status="ready", stage="ready", progress=100,
                            target=H3_PARTITION, message=f"{H3_PARTITION.upper()} 模型已就绪",
                        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        result["backend"] = "unavailable"
        result["message"] = str(exc)
    # /health 200 不等于能干活：rank 崩溃后 API 层仍然存活，但任何请求都会
    # 以 "no live replica" 失败。这种僵死只能从日志识别，必须单独告知前端，
    # 否则用户会一直提交、一直失败，却看到"引擎在线"。
    dead_marker = await asyncio.to_thread(_backend_engine_dead) if H3_PROVIDER.is_local else None
    if dead_marker:
        result["backend"] = "dead"
        result["engine_dead"] = True
        result["message"] = (
            "后端推理引擎已崩溃（日志出现 “" + dead_marker + "”）。"
            "仅重启服务可恢复：tmux kill-session -t h3-api 后重新执行 ./run_h3.sh。"
        )
    state = dict(model_state)
    if state.get("status") == "loading" and state.get("started_at"):
        elapsed = max(0, int(time.time()) - int(state["started_at"]))
        # H3 does not expose byte-level loader progress. Use explicit startup
        # phases and only a coarse range until /load confirms readiness.
        if result.get("backend") != "ok":
            if elapsed < 600:
                state["stage"] = "loading_weights"
                state["progress"] = 35
                phase = "正在加载权重"
            else:
                state["stage"] = "initializing"
                state["progress"] = 85
                phase = "正在初始化推理引擎"
            state["message"] = f"{phase}（已等待 {elapsed // 60} 分钟）"
    result["provider"] = H3_PROVIDER.mode
    result["model_name"] = H3_PROVIDER.model
    result["partition"] = str(state.get("partition") or H3_PARTITION)
    result["supported_tasks"] = (
        sorted(PARTITION_TASKS.get(result["partition"], set()))
        if H3_PROVIDER.is_local and state.get("status") == "ready"
        else ([] if H3_PROVIDER.is_local else sorted(H3_PROVIDER.tasks))
    )
    result["model"] = state
    if state.get("status") == "loading":
        result["backend"] = "switching"
        result["message"] = state.get("message") or "正在切换模型"
    elif state.get("status") == "error":
        result["backend"] = "unavailable"
        result["message"] = state.get("message") or "模型切换失败"
    return result


@app.get("/api/model")
async def get_model_state() -> dict[str, Any]:
    _sync_model_state_from_disk()
    return dict(model_state)


@app.post("/api/model/switch")
async def switch_model(request: ModelSwitchRequest) -> dict[str, Any]:
    global model_switch_task
    if not H3_PROVIDER.is_local:
        raise HTTPException(status_code=409, detail="API 模式由远程服务管理模型，工作台不支持本地分区切换")
    target = request.partition
    async with model_switch_lock:
        _sync_model_state_from_disk()
        current = str(model_state.get("partition") or H3_PARTITION)
        status = str(model_state.get("status") or "ready")
        if status == "loading":
            if str(model_state.get("target")) == target:
                return dict(model_state)
            raise HTTPException(status_code=409, detail="已有模型切换正在进行，请等待完成")
        if target == current and status == "ready":
            return dict(model_state)
        active = [record for record in jobs.values() if record.get("status") not in TERMINAL_STATUSES]
        if active:
            raise HTTPException(status_code=409, detail="当前仍有生成任务运行，请等待完成或先在作品与队列中取消任务")
        await _set_model_state(
            partition=target, target=target, status="loading", stage="stopping",
            progress=3, message=f"准备切换到 {target.upper()} 模型…", error=None,
            started_at=int(time.time()),
        )
        try:
            model_switch_task = asyncio.create_task(
                asyncio.to_thread(
                    subprocess.Popen,
                    _model_switch_command(target),
                    cwd=str(ROOT),
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        except (OSError, RuntimeError) as exc:
            await _set_model_state(status="error", stage="error", progress=0, error=str(exc), message=f"模型切换启动失败：{exc}")
            raise HTTPException(status_code=500, detail=f"模型切换启动失败：{exc}") from exc
    return dict(model_state)


def _inspect_image(name: str, payload: bytes) -> tuple[int, int]:
    """按 H3 的参考图约束校验，返回 (width, height)。"""
    if len(payload) > H3_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"{name}：图片超过 30 MB 上限")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = (image.format or "").lower()
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"{name}：无法解析为图片（{exc}）") from exc
    if image_format not in H3_IMAGE_FORMATS:
        allowed = "、".join(sorted(H3_IMAGE_FORMATS))
        raise HTTPException(status_code=400, detail=f"{name}：格式 {image_format or '未知'} 不受支持，需为 {allowed}")
    if min(width, height) < H3_IMAGE_MIN_EDGE or max(width, height) > H3_IMAGE_MAX_EDGE:
        raise HTTPException(
            status_code=400,
            detail=f"{name}：尺寸 {width}x{height} 超出 [{H3_IMAGE_MIN_EDGE}, {H3_IMAGE_MAX_EDGE}] 像素范围",
        )
    ratio = width / height
    if not H3_IMAGE_MIN_RATIO <= ratio <= H3_IMAGE_MAX_RATIO:
        raise HTTPException(
            status_code=400,
            detail=f"{name}：宽高比 {ratio:.3f} 超出 [{H3_IMAGE_MIN_RATIO}, {H3_IMAGE_MAX_RATIO}] 范围",
        )
    return width, height


def _assert_task_supported(task: str) -> None:
    """当前后端分区不支持这个任务就直接拒绝。

    两个 DiT 分区不能同时加载（合起来 263GB > 251GB 物理内存），所以
    t2va/fl2va 与 ref2va 互斥。提前拒绝比让用户等到 GPU 才报错好得多。
    """
    _sync_model_state_from_disk()
    if not H3_PROVIDER.is_local:
        if H3_PROVIDER.supports(task):
            return
        raise HTTPException(status_code=409, detail=f"当前 API 配置不支持 {task} 任务")
    partition = str(model_state.get("partition") or H3_PARTITION)
    if model_state.get("status") == "loading":
        raise HTTPException(status_code=503, detail="模型正在切换和加载，请等待顶部进度达到 100% 后再提交")
    supported = PARTITION_TASKS.get(partition, set())
    if task in supported:
        return
    other = "fl2va" if partition == "ref2va" else "ref2va"
    raise HTTPException(
        status_code=409,
        detail=(
            f"当前后端加载的是 {partition.upper()} 分区，不支持 {task} 任务。"
            f"两个分区合计约 263GB，超过本机 251GB 内存，无法同时加载。"
            f"需要该任务请切换分区：先停 h3-api，再执行 H3_PARTITION={other} ./run_h3.sh（约 13 分钟）。"
        ),
    )


def _assert_engine_alive() -> None:
    """引擎僵死时不要再往里塞任务 —— 每个都会失败，还会刷满失败记录。"""
    if not H3_PROVIDER.is_local:
        return
    marker = _backend_engine_dead()
    if marker:
        raise HTTPException(
            status_code=503,
            detail=(
                "后端推理引擎已崩溃，无法接受新任务。"
                "请重启：tmux kill-session -t h3-api，然后 ./run_h3.sh（约 13 分钟）。"
            ),
        )


@app.post("/api/generate")
async def generate(
    payload: str = Form(...),
    images: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    _assert_engine_alive()
    try:
        request = GenerateRequest.model_validate_json(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        raise HTTPException(status_code=422, detail=f"{field or '参数'}：{first.get('msg', '无效')}") from exc

    _assert_task_supported(request.task)
    uploads = [item for item in images if item.filename]
    request.validate_contract(len(uploads))

    # 校验并读入参考图，同时算出实际画布。fl2va 三种模式的比例都取自第一张图
    # （_resolve_minimax_h3_aspect_ratio 只看 images[0]），用户选的 aspect_ratio 被忽略。
    image_payloads: list[tuple[str, bytes]] = []
    canvas_width, canvas_height = canvas_for_ratio_name(request.aspect_ratio)
    labels = request.frame_labels
    for index, upload in enumerate(uploads):
        label = labels[index] if index < len(labels) else f"图 {index + 1}"
        data = await upload.read()
        width, height = _inspect_image(label, data)
        if index == 0:
            canvas_width, canvas_height = h3_canvas(width / height)
        image_payloads.append((upload.filename or f"{label}.png", data))

    # 只发 H3 真正读取的字段。刻意不发 width/height/size/fps：省略它们时
    # pipeline 会走 _resolve_output_canvas 官方配方（含最大像素钳制），而给了
    # width/height 会绕过那条路径，fl2va 下还会把参考图拉伸到错误尺寸。
    extra_params: dict[str, Any] = {"task": request.task}
    if request.resume_checkpoint_key:
        extra_params["resume_checkpoint_key"] = request.resume_checkpoint_key
    if request.task == "fl2va":
        extra_params["frame_indices"] = H3_FL2VA_MODES[request.fl2va_mode]["frame_indices"]
    if request.audio_flow_shift is not None:
        extra_params["audio_flow_shift"] = request.audio_flow_shift
    if request.force_refresh_step_hint is not None:
        extra_params["force_refresh_step_hint"] = request.force_refresh_step_hint
    if request.force_refresh_step_policy is not None:
        extra_params["force_refresh_step_policy"] = request.force_refresh_step_policy

    fields: dict[str, str] = {
        "prompt": request.prompt,
        "seconds": str(request.seconds),
        "quality": request.quality,
        "generate_sound": str(request.generate_sound).lower(),
        "extra_params": json.dumps(extra_params, ensure_ascii=False),
    }
    if request.task == "t2va":
        fields["aspect_ratio"] = request.aspect_ratio
    optional: dict[str, Any] = {
        "seed": request.seed,
        "num_inference_steps": request.num_inference_steps,
        "flow_shift": request.flow_shift,
        "sound_duration": request.sound_duration,
    }
    if request.enable_frame_interpolation:
        fields["enable_frame_interpolation"] = "true"
        optional["frame_interpolation_exp"] = request.frame_interpolation_exp
        optional["frame_interpolation_scale"] = request.frame_interpolation_scale
    fields.update({key: str(value) for key, value in optional.items() if value is not None})

    multipart: list[tuple[str, Any]] = [(key, (None, value)) for key, value in fields.items()]
    for filename, data in image_payloads:
        multipart.append(("input_references", (filename, data, "application/octet-stream")))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{H3_BASE_URL}/v1/videos", headers=_headers(), files=multipart)
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"H3 后端不可用：{exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=_error_text(response))

    result = response.json()
    job_id = result.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail="H3 返回结果中没有任务 ID")

    # Keep an exact, private copy of the request inputs. H3 itself exposes no
    # queue-priority API, so moving a waiting task requires cancelling and
    # resubmitting the waiting suffix in the chosen order.
    asset_files: list[str] = []
    if image_payloads:
        asset_dir = QUEUE_ASSET_ROOT / job_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        for index, (filename, data) in enumerate(image_payloads):
            suffix = Path(filename).suffix.lower() or ".img"
            asset_path = asset_dir / f"{index:02d}{suffix}"
            asset_path.write_bytes(data)
            asset_files.append(str(asset_path.relative_to(ROOT)))

    frames = request.frames
    await _update_job(
        job_id,
        status=str(result.get("status", "queued")).lower(),
        prompt=request.prompt,
        task=request.task,
        fl2va_mode=request.fl2va_mode if request.task == "fl2va" else None,
        aspect_ratio=request.aspect_ratio if request.task == "t2va" else f"{canvas_width}:{canvas_height}",
        seconds=request.seconds,
        actual_seconds=round(frames / H3_FPS, 3),
        frames=frames,
        size=f"{canvas_width}x{canvas_height}",
        pixels=canvas_width * canvas_height,
        quality=request.quality,
        steps=request.steps,
        seed=request.seed,
        generate_sound=request.generate_sound,
        reference_images=len(image_payloads),
        request_snapshot=request.model_dump(mode="json"),
        asset_files=asset_files,
        created_at=int(result.get("created_at") or time.time()),
        started_at=None,
        message=None,
    )
    _start_watcher(job_id)
    return _decorate_job(jobs[job_id], _eta_model())


def _reference_kind(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    suffix = Path(upload.filename or "").suffix.lower()
    if content_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}:
        return "image"
    if content_type.startswith("audio/") or suffix in {".wav", ".mp3"}:
        return "audio"
    return "video"


def _video_has_audio(data: bytes, suffix: str) -> bool:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(temporary)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reference_mapping_line(item: dict[str, Any]) -> str:
    subject = f"，主体是{item['subject']}" if item.get("subject") else ""
    description = f"。{item['description']}" if item.get("description") else ""
    priority = {"primary": "这是主要约束，必须优先保持", "support": "作为辅助约束", "background": "只约束背景，不得覆盖主体身份"}[item["priority"]]
    return f"{item['model_token']} 是“{item['label']}”，用途：{item['role']}{subject}。{priority}{description}"


REF_VIDEO_MIN_FPS = 23.976
REF_VIDEO_MAX_FPS = 60.0
REF_VIDEO_MIN_EDGE = 256
REF_VIDEO_MAX_EDGE = 5760
REF_VIDEO_MIN_RATIO = 0.4
REF_VIDEO_MAX_RATIO = 2.5
REF_VIDEO_CONTAINERS = {"mp4", "mov", "m4a", "3gp", "3g2", "mj2", "isom", "iso2", "avc1", "mp41", "mp42", "qt"}
REF_VIDEO_CODECS = {"h264", "hevc"}
REF_AUDIO_CODECS = {"aac", "mp3"}


def _probe_reference_video(path: Path) -> dict[str, Any]:
    """用 ffprobe 读出校验所需的元信息。"""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
            "-show_entries", "format=format_name,duration,size",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"无法解析视频：{completed.stderr.strip()[:200]}")
    data = json.loads(completed.stdout or "{}")
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("文件里没有视频轨")
    fps_text = str(video.get("r_frame_rate") or "0/1")
    numerator, _, denominator = fps_text.partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 0.0
    fmt = data.get("format") or {}
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": fps,
        "duration": float(fmt.get("duration") or 0.0),
        "container": {part.strip().lower() for part in str(fmt.get("format_name") or "").split(",") if part.strip()},
        "video_codec": str(video.get("codec_name") or "").lower(),
        "audio_codecs": [str(item.get("codec_name") or "").lower() for item in streams if item.get("codec_type") == "audio"],
        "size_bytes": int(fmt.get("size") or path.stat().st_size),
    }


def _reference_video_problems(path: Path, label: str) -> list[str]:
    """把参考视频不合规的地方全部列出来（不合规就别送进 GPU）。

    worker 侧校验失败会拖垮 NCCL 并杀死整个 stage，所以宁可在这里啰嗦。
    """
    try:
        meta = _probe_reference_video(path)
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return [f"{label}：{exc}"]

    problems: list[str] = []
    width, height = meta["width"], meta["height"]
    if not width or not height:
        problems.append(f"{label}：读不到画面尺寸")
    else:
        if min(width, height) < REF_VIDEO_MIN_EDGE or max(width, height) > REF_VIDEO_MAX_EDGE:
            problems.append(f"{label}：尺寸 {width}x{height} 超出 [256, 5760] 像素范围")
        ratio = width / height
        if not REF_VIDEO_MIN_RATIO <= ratio <= REF_VIDEO_MAX_RATIO:
            problems.append(f"{label}：宽高比 {ratio:.2f} 超出 [0.4, 2.5]")
    if not REF_VIDEO_MIN_FPS <= meta["fps"] <= REF_VIDEO_MAX_FPS:
        problems.append(f"{label}：帧率 {meta['fps']:.3f} 超出 [23.976, 60]")
    duration = meta["duration"]
    if not math.isfinite(duration) or not REF_VIDEO_MIN_SECONDS <= duration <= REF_VIDEO_MAX_SECONDS:
        problems.append(f"{label}：时长 {duration:.3f} 秒超出 [2, 15] 秒")
    if meta["size_bytes"] > 50 * 1024 * 1024:
        problems.append(f"{label}：体积超过 50MB 上限")
    if not meta["container"] & REF_VIDEO_CONTAINERS:
        problems.append(f"{label}：容器必须是 MP4 或 MOV")
    if meta["video_codec"] not in REF_VIDEO_CODECS:
        problems.append(f"{label}：视频编码必须是 H.264 或 H.265，当前 {meta['video_codec'] or '未知'}")
    invalid_audio = [codec for codec in meta["audio_codecs"] if codec not in REF_AUDIO_CODECS]
    if invalid_audio:
        problems.append(f"{label}：音轨编码必须是 AAC 或 MP3，当前 {invalid_audio[0]}")
    return problems


@app.post("/api/ref2va")
async def generate_ref2va(
    payload: str = Form(...),
    references: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    _assert_engine_alive()
    _assert_task_supported("ref2va")
    try:
        request = Ref2VARequest.model_validate_json(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        raise HTTPException(status_code=422, detail=f"{field or '参数'}：{first.get('msg', '无效')}") from exc

    uploads = [item for item in references if item.filename]
    if len(request.references_meta) != len(uploads):
        raise HTTPException(status_code=400, detail="素材说明与上传文件数量不一致，请重新检查 Reference Board")
    for index, (upload, meta) in enumerate(zip(uploads, request.references_meta, strict=True), 1):
        if _reference_kind(upload) != meta.kind:
            raise HTTPException(status_code=400, detail=f"第 {index} 个素材类型与说明不一致：{upload.filename}")
    enabled_pairs = [(upload, meta) for upload, meta in zip(uploads, request.references_meta, strict=True) if meta.enabled]
    uploads = [item[0] for item in enabled_pairs]
    enabled_meta = [item[1] for item in enabled_pairs]
    if not uploads:
        raise HTTPException(status_code=400, detail="Ref2VA 至少需要一个图片、视频或音频参考")
    if len(uploads) > 12:
        raise HTTPException(status_code=400, detail="Ref2VA 最多接受 12 个参考素材")

    counts = {"image": 0, "video": 0, "audio": 0}
    media_payloads: list[tuple[str, bytes, str, str, ReferenceMeta, bool]] = []
    limits = {"image": 30 * 1024 * 1024, "video": 50 * 1024 * 1024, "audio": 15 * 1024 * 1024}
    allowed = {
        "image": {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"},
        "video": {".mp4", ".mov"},
        "audio": {".wav", ".mp3"},
    }
    for upload, meta in zip(uploads, enabled_meta, strict=True):
        kind = _reference_kind(upload)
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in allowed[kind]:
            raise HTTPException(status_code=400, detail=f"{kind} 素材格式不支持：{upload.filename}")
        data = await upload.read()
        if len(data) > limits[kind]:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} 超过 {limits[kind] // (1024 * 1024)}MB 上限",
            )
        if kind == "image":
            _inspect_image(upload.filename or "参考图", data)
        has_audio = kind == "video" and await asyncio.to_thread(_video_has_audio, data, suffix)
        counts[kind] += 1
        media_payloads.append((upload.filename or f"reference{suffix}", data, upload.content_type or "application/octet-stream", kind, meta, has_audio))

    # 视频参考必须在这里验完 —— worker 侧校验失败会拖垮 NCCL 并杀死整个 stage
    # （实测 1 秒的参考视频就让后端永久死亡，只能重启 13 分钟）。
    video_problems: list[str] = []
    for filename, data, _, kind, _, _ in media_payloads:
        if kind != "video":
            continue
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            probe_path = Path(handle.name)
        try:
            video_problems.extend(await asyncio.to_thread(_reference_video_problems, probe_path, filename))
        finally:
            probe_path.unlink(missing_ok=True)
    if video_problems:
        raise HTTPException(status_code=400, detail="参考视频不符合 Ref2VA 要求：" + "；".join(video_problems))

    if counts["image"] > 9:
        raise HTTPException(status_code=400, detail="Ref2VA 最多接受 9 张图片")
    if counts["video"] > 3:
        raise HTTPException(status_code=400, detail="Ref2VA 最多接受 3 个视频")
    if counts["audio"] > 3:
        raise HTTPException(status_code=400, detail="Ref2VA 最多接受 3 个独立音频")

    picture_no = video_no = audio_no = 0
    reference_manifest: list[dict[str, Any]] = []
    for _, _, _, kind, meta, has_audio in media_payloads:
        if kind == "image":
            picture_no += 1
            model_token = f"<Picture {picture_no}>"
        elif kind == "video":
            video_no += 1
            model_token = f"<Video {video_no}>"
        else:
            audio_no += 1
            model_token = f"<Audio {audio_no}>"
        audio_token = None
        if kind == "video" and has_audio:
            audio_no += 1
            audio_token = f"<Audio {audio_no}>"
        reference_manifest.append({**meta.model_dump(mode="json"), "model_token": model_token, "audio_token": audio_token, "has_audio": has_audio})
    mapping_lines = [_reference_mapping_line(item) for item in reference_manifest]
    for item in reference_manifest:
        if item.get("audio_token"):
            mapping_lines.append(f"{item['audio_token']} 是 {item['model_token']} 自带的声音轨道，只用于该视频的声音与节奏连续性。")
    effective_prompt = "参考素材身份映射（必须严格遵守）：\n" + "\n".join(mapping_lines) + "\n\n生成要求：\n" + request.prompt

    extra_params: dict[str, Any] = {"task": "ref2va"}
    if request.audio_flow_shift is not None:
        extra_params["audio_flow_shift"] = request.audio_flow_shift
    fields: dict[str, str] = {
        "prompt": effective_prompt,
        "seconds": str(request.seconds),
        "aspect_ratio": request.aspect_ratio,
        "quality": request.quality,
        "generate_sound": str(request.generate_sound).lower(),
        "extra_params": json.dumps(extra_params, ensure_ascii=False),
    }
    optional: dict[str, Any] = {
        "seed": request.seed,
        "num_inference_steps": request.num_inference_steps,
        "flow_shift": request.flow_shift,
        "sound_duration": request.sound_duration,
        "start_time_seconds": request.start_time_seconds,
    }
    fields.update({key: str(value) for key, value in optional.items() if value is not None})
    multipart: list[tuple[str, Any]] = [(key, (None, value)) for key, value in fields.items()]
    for filename, data, content_type, _, _, _ in media_payloads:
        multipart.append(("input_references", (filename, data, content_type)))

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{H3_BASE_URL}/v1/videos", headers=_headers(), files=multipart)
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"H3 Ref2VA 后端不可用：{exc}") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=_error_text(response))
    result = response.json()
    job_id = result.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail="H3 返回结果中没有任务 ID")

    asset_dir = QUEUE_ASSET_ROOT / job_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_files: list[str] = []
    manifest: list[dict[str, Any]] = []
    for index, ((filename, data, _, kind, _, _), mapped) in enumerate(zip(media_payloads, reference_manifest, strict=True)):
        suffix = Path(filename).suffix.lower()
        asset_path = asset_dir / f"{index:02d}{suffix}"
        asset_path.write_bytes(data)
        asset_files.append(str(asset_path.relative_to(ROOT)))
        manifest.append({**mapped, "name": filename, "kind": kind, "asset_index": index, "path": str(asset_path.relative_to(ROOT))})

    width, height = canvas_for_ratio_name(request.aspect_ratio)
    await _update_job(
        job_id,
        status=str(result.get("status", "queued")).lower(),
        prompt=request.prompt,
        effective_prompt=effective_prompt,
        task="ref2va",
        aspect_ratio=request.aspect_ratio,
        seconds=request.seconds,
        actual_seconds=round(request.frames / H3_FPS, 3),
        frames=request.frames,
        size=f"{width}x{height}",
        pixels=width * height,
        quality=request.quality,
        steps=request.steps,
        seed=request.seed,
        generate_sound=request.generate_sound,
        reference_images=counts["image"],
        reference_videos=counts["video"],
        reference_audios=counts["audio"],
        reference_manifest=manifest,
        request_snapshot=request.model_dump(mode="json"),
        asset_files=asset_files,
        created_at=int(result.get("created_at") or time.time()),
        started_at=None,
        message=None,
    )
    _start_watcher(job_id)
    return _decorate_job(jobs[job_id], _eta_model())


def _request_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    snapshot = record.get("request_snapshot")
    if isinstance(snapshot, dict):
        return dict(snapshot)
    # Compatibility for tasks submitted before queue assets were introduced.
    return {
        "prompt": record.get("prompt") or "",
        "task": record.get("task") or "t2va",
        "fl2va_mode": record.get("fl2va_mode") or "first",
        "aspect_ratio": record.get("aspect_ratio")
        if record.get("task") == "t2va" and record.get("aspect_ratio") in H3_ASPECT_RATIOS
        else "16:9",
        "seconds": record.get("seconds") or 5,
        "quality": record.get("quality") or "lossless",
        "seed": record.get("seed"),
        "num_inference_steps": record.get("steps") or H3_DEFAULT_STEPS,
        "generate_sound": record.get("generate_sound", True),
    }


def _saved_assets(record: dict[str, Any]) -> list[Path]:
    paths = []
    for relative in record.get("asset_files") or []:
        path = ROOT / str(relative)
        if path.is_file():
            paths.append(path)
    # The current cat batch predates input persistence and all uses the single
    # project reference image. This fallback only applies to one-image jobs.
    if not paths and record.get("reference_images") == 1 and LEGACY_REFERENCE_IMAGE.is_file():
        paths.append(LEGACY_REFERENCE_IMAGE)
    return paths


def _director_shot_from_text(text: str, index: int, seconds: int) -> dict[str, Any]:
    cleaned = text.strip()
    title = cleaned.splitlines()[0][:42] if cleaned else f"镜头 {index + 1}"
    return {
        "id": f"shot_{uuid.uuid4().hex[:10]}",
        "order": index,
        "title": title,
        "prompt": cleaned,
        "seconds": seconds,
        "seed_offset": index,
        "continuity": "auto",
        "character_ids": [],
        "location_id": None,
        "scene_asset_id": None,
        "asset_ids": [],
        "start_state": "",
        "end_state": "",
        "camera": "",
        "sound": "",
        "trim_start": 0.0,
        "trim_end": None,
        "audio_gain": 1.0,
        "review_note": "",
        "status": "draft",
        "job_id": None,
        "filename": None,
        "message": None,
    }


def _split_shot_plan(plan: str) -> list[str]:
    blocks = [item.strip() for item in re.split(r"\n\s*\n|(?=【?\d{1,2}:\d{2})", plan) if item.strip()]
    if len(blocks) <= 1:
        blocks = [item.strip(" -\t") for item in plan.splitlines() if item.strip(" -\t")]
    return blocks


@app.post("/api/director/projects")
async def create_director_project(request: DirectorProjectRequest) -> dict[str, Any]:
    project_id = f"director_{uuid.uuid4().hex[:12]}"
    now = int(time.time())
    shot_texts = _split_shot_plan(request.shot_plan)
    project = {
        "id": project_id,
        "title": request.title.strip(),
        "synopsis": request.synopsis.strip(),
        "visual_bible": request.visual_bible.strip(),
        "identity_bible": request.identity_bible.strip(),
        "style_bible": request.style_bible.strip(),
        "aspect_ratio": request.aspect_ratio,
        "default_seconds": request.default_seconds,
        "quality": request.quality,
        "seed": request.seed,
        "overlap_seconds": request.overlap_seconds,
        "generate_sound": request.generate_sound,
        "sound_mode": request.sound_mode,
        "review_mode": request.review_mode,
        "auto_approve": request.review_mode == "pipeline",
        "status": "draft",
        "assets": [],
        "entities": [],
        "shots": [
            _director_shot_from_text(text, index, request.default_seconds)
            for index, text in enumerate(shot_texts)
        ],
        "output_filename": None,
        "created_at": now,
        "updated_at": now,
    }
    async with director_lock:
        director_projects[project_id] = project
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.get("/api/director/projects")
async def list_director_projects() -> list[dict[str, Any]]:
    return [
        _director_project_view(project)
        for project in sorted(
            director_projects.values(),
            key=lambda item: item.get("updated_at", 0),
            reverse=True,
        )
    ]


@app.get("/api/director/projects/{project_id}")
async def get_director_project(project_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    return _director_project_view(project)


@app.get("/api/director/projects/{project_id}/shots/{shot_id}/video")
async def director_shot_video(project_id: str, shot_id: str) -> FileResponse:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project.get("shots", []) if item.get("id") == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="导演镜头不存在")
    filename = shot.get("filename")
    if not filename:
        raise HTTPException(status_code=404, detail="镜头尚未生成视频")
    path = _safe_output_video_path(str(filename))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="镜头视频文件不存在")
    return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")


@app.get("/api/director/projects/{project_id}/output")
async def director_output_video(project_id: str) -> FileResponse:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    filename = project.get("output_filename")
    if not filename:
        raise HTTPException(status_code=404, detail="导演成片尚未导出")
    path = _safe_output_video_path(str(filename))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="导演成片文件不存在")
    return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")


@app.put("/api/director/projects/{project_id}")
async def update_director_project(project_id: str, request: DirectorProjectUpdate) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    if project.get("status") == "running":
        raise HTTPException(status_code=409, detail="运行中的导演项目不能修改总设定，请先停止生成")
    changes = request.model_dump(exclude_unset=True)
    if "auto_approve" in changes and "review_mode" not in changes:
        changes["review_mode"] = "pipeline" if changes["auto_approve"] else "review_gate"
    if "review_mode" in changes:
        changes["auto_approve"] = changes["review_mode"] == "pipeline"
    if "sound_mode" in changes:
        changes["generate_sound"] = changes["sound_mode"] == "native"
    elif "generate_sound" in changes:
        changes["sound_mode"] = "native" if changes["generate_sound"] else "off"
    for key, value in changes.items():
        if value is not None:
            project[key] = value.strip() if isinstance(value, str) else value
    if _project_review_mode(project) == "pipeline":
        # Switching an existing project out of review-gate should release any
        # already-completed shot that was waiting for an approval click.
        for shot in project.get("shots", []):
            if shot.get("status") == "awaiting_review" and shot.get("filename"):
                shot["status"] = "completed"
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/normalize-test-duration")
async def normalize_director_test_duration(project_id: str) -> dict[str, Any]:
    """Set every draft/test shot to the minimum valid 4-second duration."""
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    if project.get("status") == "running":
        raise HTTPException(status_code=409, detail="请先停止导演项目")
    for shot in project.get("shots", []):
        shot["seconds"] = H3_MIN_SECONDS
        shot["status"] = "draft"
        shot["job_id"] = None
        shot["filename"] = None
        shot["message"] = None
    project["default_seconds"] = H3_MIN_SECONDS
    project["output_filename"] = None
    project["status"] = "paused"
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/assets")
async def upload_director_assets(
    project_id: str,
    files: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    uploads = [item for item in files if item.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="请选择导演参考素材")
    project_root = DIRECTOR_ASSET_ROOT / project_id / "references"
    project_root.mkdir(parents=True, exist_ok=True)
    added: list[dict[str, Any]] = []
    for upload in uploads:
        kind = _reference_kind(upload)
        suffix = Path(upload.filename or "").suffix.lower()
        allowed = {
            "image": {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"},
            "video": {".mp4", ".mov"},
            "audio": {".wav", ".mp3"},
        }
        if suffix not in allowed[kind]:
            raise HTTPException(status_code=400, detail=f"不支持的导演素材：{upload.filename}")
        data = await upload.read()
        limits = {"image": 30, "video": 50, "audio": 15}
        if len(data) > limits[kind] * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{upload.filename} 超过 {limits[kind]}MB")
        if kind == "image":
            _inspect_image(upload.filename or "导演参考图", data)
        asset_id = f"asset_{uuid.uuid4().hex[:10]}"
        path = project_root / f"{asset_id}{suffix}"
        path.write_bytes(data)
        asset = {
            "id": asset_id,
            "kind": kind,
            "name": upload.filename,
            "path": str(path.relative_to(DIRECTOR_ASSET_ROOT / project_id)),
            "size_bytes": len(data),
            "role": "identity" if kind == "image" else ("motion" if kind == "video" else "sound"),
            "subject": "",
            "description": "",
            "priority": "support",
            "enabled": True,
            "created_at": int(time.time()),
        }
        project["assets"].append(asset)
        added.append(asset)
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return {**_director_project_view(project), "added_assets": added}


@app.get("/api/director/projects/{project_id}/assets/{asset_id}")
async def director_asset(project_id: str, asset_id: str) -> FileResponse:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    asset = next((item for item in project.get("assets", []) if item.get("id") == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="导演素材不存在")
    path = _director_asset_path(project_id, str(asset["path"]))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="导演素材文件不存在")
    return FileResponse(path)


@app.put("/api/director/projects/{project_id}/assets/{asset_id}")
async def update_director_asset(project_id: str, asset_id: str, request: DirectorAssetUpdate) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    asset = next((item for item in project.get("assets", []) if item.get("id") == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="导演素材不存在")
    asset.update(request.model_dump(mode="json"))
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.delete("/api/director/projects/{project_id}/assets/{asset_id}")
async def delete_director_asset(project_id: str, asset_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    if project.get("status") == "running":
        raise HTTPException(status_code=409, detail="运行中的导演项目不能删除素材，请先停止生成")
    asset = next((item for item in project.get("assets", []) if item.get("id") == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="导演素材不存在")
    project["assets"] = [item for item in project.get("assets", []) if item.get("id") != asset_id]
    for entity in project.get("entities", []):
        entity["asset_ids"] = [item for item in entity.get("asset_ids", []) if item != asset_id]
    for shot in project.get("shots", []):
        shot["asset_ids"] = [item for item in shot.get("asset_ids", []) if item != asset_id]
        if shot.get("scene_asset_id") == asset_id:
            shot["scene_asset_id"] = None
    path = _director_asset_path(project_id, str(asset.get("path") or ""))
    path.unlink(missing_ok=True)
    project["output_filename"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/entities")
async def add_director_entity(project_id: str, request: DirectorEntityRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    valid_assets = {item["id"] for item in project.get("assets", [])}
    if any(asset_id not in valid_assets for asset_id in request.asset_ids):
        raise HTTPException(status_code=400, detail="实体绑定了不存在的素材")
    entity = {"id": f"entity_{uuid.uuid4().hex[:10]}", **request.model_dump(mode="json"), "created_at": int(time.time())}
    project.setdefault("entities", []).append(entity)
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.put("/api/director/projects/{project_id}/entities/{entity_id}")
async def update_director_entity(project_id: str, entity_id: str, request: DirectorEntityRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    entity = next((item for item in project.get("entities", []) if item.get("id") == entity_id), None)
    if entity is None:
        raise HTTPException(status_code=404, detail="导演实体不存在")
    valid_assets = {item["id"] for item in project.get("assets", [])}
    if any(asset_id not in valid_assets for asset_id in request.asset_ids):
        raise HTTPException(status_code=400, detail="实体绑定了不存在的素材")
    entity.update(request.model_dump(mode="json"))
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.delete("/api/director/projects/{project_id}/entities/{entity_id}")
async def delete_director_entity(project_id: str, entity_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    if project.get("status") == "running":
        raise HTTPException(status_code=409, detail="运行中的导演项目不能删除角色或场景，请先停止生成")
    before = len(project.get("entities", []))
    project["entities"] = [item for item in project.get("entities", []) if item.get("id") != entity_id]
    if len(project["entities"]) == before:
        raise HTTPException(status_code=404, detail="导演实体不存在")
    for shot in project.get("shots", []):
        shot["character_ids"] = [item for item in shot.get("character_ids", []) if item != entity_id]
        if shot.get("location_id") == entity_id:
            shot["location_id"] = None
    project["output_filename"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


def _states_align(previous_end: str, current_start: str) -> bool:
    """判断相邻镜头的结束/开始状态是否描述同一个画面。

    原实现要求两段文字完全相等，但这是人手写的自然语言，逐字相同不现实，
    结果每一对相邻镜头都刷一条假警告。改成按关键词重叠度判断：
    取中文双字词做集合，重叠超过三成就认为讲的是同一个落幅。
    """
    def keywords(text: str) -> set[str]:
        cleaned = re.sub(r"[^\u4e00-\u9fff]+", "", text)
        return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}

    before, after = keywords(previous_end), keywords(current_start)
    if not before or not after:
        return True
    overlap = len(before & after) / min(len(before), len(after))
    return overlap >= 0.3


def _director_continuity_report(project: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    entities = {item["id"]: item for item in project.get("entities", [])}
    assets = {item["id"]: item for item in project.get("assets", []) if item.get("enabled", True)}
    shots = sorted(project.get("shots", []), key=lambda item: item.get("order", 0))
    index = next((i for i, item in enumerate(shots) if item.get("id") == shot.get("id")), 0)
    for entity_id in shot.get("character_ids", []):
        entity = entities.get(entity_id)
        if entity is None:
            issues.append({"level": "error", "message": "镜头绑定了已不存在的角色"})
        elif not any(asset_id in assets for asset_id in entity.get("asset_ids", [])):
            issues.append({"level": "error", "message": f"角色“{entity['name']}”没有启用的身份参考素材"})
    if shot.get("location_id") and shot["location_id"] not in entities:
        issues.append({"level": "error", "message": "镜头绑定的场景已不存在"})
    scene_asset_id = str(shot.get("scene_asset_id") or "")
    scene_asset = assets.get(scene_asset_id) if scene_asset_id else None
    # 场景图是显式切换/锁定场景的可选项。没有选择时，连续镜头直接继承上一镜
    # 的尾帧（FL2VA）或尾部视频（Ref2VA），不应因为缺少独立场景图阻塞流水线。
    if scene_asset_id and scene_asset is None:
        issues.append({"level": "error", "message": "镜头绑定的场景参考图不存在或已停用"})
    elif scene_asset is not None and scene_asset.get("kind") != "image":
        issues.append({"level": "error", "message": "场景参考必须是图片"})
    if any(asset_id not in assets for asset_id in shot.get("asset_ids", [])):
        issues.append({"level": "error", "message": "镜头绑定了不存在或已停用的素材"})
    uses_previous = index > 0 and shot.get("continuity") in {"auto", "previous_video"}
    if uses_previous:
        previous = shots[index - 1]
        if not previous.get("end_state") or not shot.get("start_state"):
            issues.append({"level": "warning", "message": "相邻镜头缺少结束/开始状态，无法人工核对动作衔接"})
        elif not _states_align(previous["end_state"], shot["start_state"]):
            issues.append({"level": "warning", "message": f"上一镜头结束“{previous['end_state']}”与本镜头开始“{shot['start_state']}”描述差异较大，请确认动作能接上"})
        if previous.get("status") not in {"approved", "completed"}:
            # 时序问题，不是配置错误：调度器串行执行，上一镜跑完自然就绪。
            # 报成 error 会让刚建好的项目整片红、并挡住启动。
            gate_text = "并通过审核" if _project_review_mode(project) == "review_gate" else "完成"
            issues.append({"level": "pending", "message": f"等待上一镜头生成{gate_text}后才能开始本镜"})
    selected = set(shot.get("asset_ids", []))
    if shot.get("scene_asset_id"):
        selected.add(str(shot["scene_asset_id"]))
    for entity_id in shot.get("character_ids", []):
        selected.update(entities.get(entity_id, {}).get("asset_ids", []))
    selected_assets = [assets[item] for item in selected if item in assets]
    continuity_count = 1 if uses_previous else 0
    if sum(item.get("kind") == "image" for item in selected_assets) > 9 or sum(item.get("kind") == "video" for item in selected_assets) + continuity_count > 3 or len(selected_assets) + continuity_count > 12:
        issues.append({"level": "error", "message": "本镜头参考素材超过 Ref2VA 上限"})
    seconds = int(shot.get("seconds") or 5)
    prompt_text = str(shot.get("prompt") or "")
    if seconds <= 5 and len(prompt_text) > 180:
        issues.append({"level": "warning", "message": "4–5 秒镜头动作描述过长，请精简后再生成"})
    if seconds <= 5 and len(re.findall(r"[。！？!?；;]", prompt_text)) > 2:
        issues.append({"level": "warning", "message": "短镜头包含多个事件，建议只保留一个动作和一个落幅"})
    # 每个导演镜头都必须有当前模型可用的参考输入。首镜头依靠身份/场景图，
    # 后续镜头还可以从上一镜的尾帧或尾部视频获得连续性。
    if not uses_previous and not selected_assets:
        issues.append({"level": "error", "message": "该镜头没有可用的参考素材，请绑定角色身份图或直接选择素材"})
    return {
        "ready": not any(item["level"] == "error" for item in issues),
        "submittable": not any(item["level"] in {"error", "pending"} for item in issues),
        "issues": issues,
    }


@app.get("/api/director/projects/{project_id}/shots/{shot_id}/check")
async def check_director_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project.get("shots", []) if item.get("id") == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    return _director_continuity_report(project, shot)


@app.post("/api/director/projects/{project_id}/shots")
async def add_director_shot(project_id: str, request: DirectorShotRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = _director_shot_from_text(request.prompt, len(project["shots"]), request.seconds)
    shot.update(
        title=request.title.strip() or shot["title"],
        seed_offset=request.seed_offset,
        continuity=request.continuity,
        character_ids=request.character_ids,
        location_id=request.location_id,
        scene_asset_id=request.scene_asset_id,
        asset_ids=request.asset_ids,
        start_state=request.start_state.strip(),
        end_state=request.end_state.strip(),
        camera=request.camera.strip(),
        sound=request.sound.strip(),
        trim_start=request.trim_start,
        trim_end=request.trim_end,
        audio_gain=request.audio_gain,
    )
    project["shots"].append(shot)
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.put("/api/director/projects/{project_id}/shots/{shot_id}")
async def update_director_shot(project_id: str, shot_id: str, request: DirectorShotRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project["shots"] if item["id"] == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    if shot.get("status") in {"queued", "in_progress"}:
        raise HTTPException(status_code=409, detail="正在生成的镜头不能修改")
    shot.update(
        title=request.title.strip() or "新镜头",
        prompt=request.prompt.strip(),
        seconds=request.seconds,
        seed_offset=request.seed_offset,
        continuity=request.continuity,
        character_ids=request.character_ids,
        location_id=request.location_id,
        scene_asset_id=request.scene_asset_id,
        asset_ids=request.asset_ids,
        start_state=request.start_state.strip(),
        end_state=request.end_state.strip(),
        camera=request.camera.strip(),
        sound=request.sound.strip(),
        trim_start=request.trim_start,
        trim_end=request.trim_end,
        audio_gain=request.audio_gain,
        status="draft",
        job_id=None,
        filename=None,
        message=None,
    )
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.patch("/api/director/projects/{project_id}/shots/{shot_id}/edit")
async def update_director_shot_edit(
    project_id: str,
    shot_id: str,
    request: DirectorShotEditRequest,
) -> dict[str, Any]:
    """Save trim/audio settings without invalidating an already generated shot."""
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project.get("shots", []) if item.get("id") == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    if shot.get("status") in {"queued", "in_progress"}:
        raise HTTPException(status_code=409, detail="镜头生成中，暂时不能调整剪辑参数")
    values = request.model_dump(mode="json")
    trim_start = float(values["trim_start"])
    trim_end = values.get("trim_end")
    if trim_end is not None and float(trim_end) <= trim_start:
        raise HTTPException(status_code=422, detail="出点必须晚于入点")
    shot.update(values)
    project["output_filename"] = None
    if project.get("status") == "completed":
        project["status"] = "paused"
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.delete("/api/director/projects/{project_id}/shots/{shot_id}")
async def delete_director_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project["shots"] if item["id"] == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    if shot.get("status") in {"queued", "in_progress"}:
        raise HTTPException(status_code=409, detail="正在生成的镜头不能删除，请先停止生成")
    project["shots"] = [item for item in project["shots"] if item["id"] != shot_id]
    for order, item in enumerate(project["shots"]):
        item["order"] = order
    # 删掉中间某镜后，其后镜头的连续性来源已变，必须回到 draft 重新生成，
    # 否则它们会继续沿用一条已经不存在的承接关系。
    removed_order = int(shot.get("order") or 0)
    for item in project["shots"]:
        if item["order"] >= removed_order and item.get("status") in {"approved", "completed", "awaiting_review"}:
            item.update(status="draft", job_id=None, filename=None, message="上游镜头被删除，需要重新生成")
    project["output_filename"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/shots/{shot_id}/move")
async def move_director_shot(project_id: str, shot_id: str, request: DirectorMoveRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shots = project["shots"]
    index = next((i for i, item in enumerate(shots) if item["id"] == shot_id), -1)
    if index < 0:
        raise HTTPException(status_code=404, detail="镜头不存在")
    target = index - 1 if request.direction == "up" else index + 1
    if 0 <= target < len(shots):
        shots[index], shots[target] = shots[target], shots[index]
    for order, shot in enumerate(shots):
        shot["order"] = order
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


def _extract_tail_clip(source: Path, destination: Path, seconds: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-sseof", f"-{seconds:.3f}", "-i", str(source),
        "-t", f"{seconds:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "无法提取上一镜头尾部")


def _extract_last_frame(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.15", "-i", str(source),
        "-frames:v", "1", "-q:v", "2", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode != 0 or not destination.is_file():
        raise RuntimeError(completed.stderr.strip() or "无法提取上一镜头最后一帧")


def _director_reference_meta(
    project: dict[str, Any],
    selected_assets: list[dict[str, Any]],
    *,
    limit_images: int,
    limit_videos: int = 3,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """把导演素材映射成 Ref2VA 的 references_meta。

    Ref2VA 的配额是图片 9 / 视频 3 / 音频 3、总数 12。后续镜头还要留一个视频
    槽位给「上一镜头尾部」，所以两个上限都由调用方按镜头位置传入。
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    images = [item for item in selected_assets if item.get("kind") == "image"][:limit_images]
    videos = [item for item in selected_assets if item.get("kind") == "video"][:limit_videos]
    audios = [item for item in selected_assets if item.get("kind") == "audio"][:3]
    # 总数兜底：9+3+3=15 可能超过 Ref2VA 的 12 个总上限，按 图→视频→音频
    # 的优先级截断（selected_assets 已按 primary/support/background 排过序）。
    for asset in (images + videos + audios)[:12]:
        pairs.append((
            asset,
            {
                "client_id": asset["id"],
                "kind": asset["kind"],
                "label": asset.get("name") or Path(str(asset["path"])).name,
                "role": asset.get("role") or "身份参考",
                "subject": asset.get("subject") or "",
                "description": asset.get("description") or "",
                "priority": asset.get("priority") or "support",
                "enabled": True,
            },
        ))
    return pairs


async def _director_submit_fl2va(
    project: dict[str, Any],
    shot: dict[str, Any],
    prompt: str,
    seed: int,
    selected_assets: list[dict[str, Any]],
    continuity_frame: Path | None = None,
) -> str:
    """Submit a director shot through the FL2VA-compatible regular endpoint.

    FL2VA has no Ref2VA multi-media reference contract, so the director keeps
    the same story/identity prompt but adapts references to one or two images.
    Subsequent shots inherit continuity through the prompt and shot states.
    """
    image_assets = [item for item in selected_assets if item.get("kind") == "image"][:2]
    if continuity_frame is not None:
        # FL2VA images are literal keyframes, not semantic reference slots.
        # The previous shot's final frame must therefore become this shot's
        # first frame. Mixing an identity portrait as frame 0 and the previous
        # tail as frame -1 would reverse the intended time semantics.
        payload_images = [(continuity_frame.name, continuity_frame.read_bytes())]
        mode = "first"
    else:
        payload_images = []
        mode = "first_last" if len(image_assets) == 2 else "first"
    if not image_assets:
        if continuity_frame is None:
            raise RuntimeError("FL2VA 导演模式至少需要一张身份参考图")
    payload = {
        "prompt": prompt,
        "task": "fl2va",
        "fl2va_mode": mode,
        "aspect_ratio": project["aspect_ratio"],
        "seconds": shot["seconds"],
        "quality": project["quality"],
        "seed": seed,
        "generate_sound": _director_sound_enabled(project),
    }
    multipart: list[tuple[str, Any]] = [("payload", (None, json.dumps(payload, ensure_ascii=False)))]
    if payload_images:
        for name, data in payload_images:
            multipart.append(("images", (name, data, "image/png")))
    else:
        for asset in image_assets:
            path = _director_asset_path(project["id"], asset["path"])
            multipart.append(("images", (path.name, path.read_bytes(), "application/octet-stream")))
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{FRONTEND_INTERNAL_URL}/api/generate", files=multipart)
    if response.status_code != 200:
        raise RuntimeError(_error_text(response))
    result = response.json()
    return str(result["id"])


async def _director_submit_shot(project: dict[str, Any], shot: dict[str, Any]) -> str:
    shots = sorted(project["shots"], key=lambda item: item["order"])
    index = next(i for i, item in enumerate(shots) if item["id"] == shot["id"])
    report = _director_continuity_report(project, shot)
    if not report["submittable"]:
        blocking = [item["message"] for item in report["issues"] if item["level"] in {"error", "pending"}]
        raise RuntimeError("连续性检查未通过：" + "；".join(blocking))
    entities = {item["id"]: item for item in project.get("entities", [])}
    # Character entities contribute identity assets. The scene is selected by
    # its dedicated image slot, so location metadata cannot silently add other
    # backgrounds to the model request.
    selected_entity_ids = list(shot.get("character_ids", []))
    prompt_parts = [_director_shot_prompt(project, shot, index)]
    uses_previous = index > 0 and shot.get("continuity") in {"auto", "previous_video"}
    if uses_previous:
        prompt_parts.append("只承接上一镜头的最后姿态、站位和运动方向；不要读取或表现其它镜头的场景。")
    prompt = "\n".join(part for part in prompt_parts if part)
    seed = int(project.get("seed") or 42) + int(shot.get("seed_offset") or index)
    selected_asset_ids = set(shot.get("asset_ids", []))
    if shot.get("scene_asset_id"):
        selected_asset_ids.add(str(shot["scene_asset_id"]))
    for entity_id in selected_entity_ids:
        selected_asset_ids.update(entities.get(entity_id, {}).get("asset_ids", []))
    selected_assets = []
    for source in project.get("assets", []):
        if source.get("id") not in selected_asset_ids or not source.get("enabled", True):
            continue
        asset = dict(source)
        if source.get("id") == shot.get("scene_asset_id"):
            asset.update(
                role="当前镜头场景",
                subject="场景背景",
                description="本镜头唯一场景依据，不得切换到其它背景",
                priority="primary",
            )
        selected_assets.append(asset)
    selected_assets.sort(
        key=lambda item: (
            0 if item.get("id") == shot.get("scene_asset_id") else 1,
            {"primary": 0, "support": 1, "background": 2}.get(item.get("priority"), 1),
        )
    )
    _sync_model_state_from_disk()
    active_partition = str(model_state.get("partition") or H3_PARTITION).lower()
    # Persist the exact source-of-truth before submitting. This snapshot is
    # returned by the director API and shown beside the editable shot card.
    execution: dict[str, Any] = {
        "partition": active_partition,
        "task": "fl2va" if active_partition == "fl2va" else "ref2va",
        "prompt": prompt,
        "seed": seed,
        "seconds": shot["seconds"],
        "quality": project["quality"],
        "sound_mode": _director_sound_mode(project),
        "generate_sound": _director_sound_enabled(project),
        "references": [],
        "continuity_source": "none",
        "scene_reference": next(
            (item.get("name") for item in selected_assets if item.get("id") == shot.get("scene_asset_id")),
            "继承上一镜头最后场景" if uses_previous else None,
        ),
    }
    shot["execution"] = execution
    if active_partition == "fl2va":
        continuity_frame: Path | None = None
        if uses_previous:
            previous_filename = shots[index - 1].get("filename")
            previous_path = _output_path(str(previous_filename or ""))
            if not previous_filename or not previous_path.is_file():
                raise RuntimeError("上一镜头尚无成片，无法提取 FL2VA 连续性首帧")
            continuity_frame = DIRECTOR_ASSET_ROOT / project["id"] / "continuity" / f"{shots[index - 1]['id']}_last.png"
            await asyncio.to_thread(_extract_last_frame, previous_path, continuity_frame)
        execution["continuity_source"] = continuity_frame.name if continuity_frame else "identity image"
        execution["references"] = [continuity_frame.name] if continuity_frame else [item.get("name") or item.get("path") for item in selected_assets if item.get("kind") == "image"][:2]
        job_id = await _director_submit_fl2va(project, shot, prompt, seed, selected_assets, continuity_frame)
        record = jobs.get(job_id, {})
        execution["job_id"] = job_id
        execution["model_prompt"] = record.get("effective_prompt") or record.get("prompt") or prompt
        execution["request_snapshot"] = record.get("request_snapshot")
        return job_id
    image_assets = [asset for asset in selected_assets if asset.get("kind") == "image"][:9]

    if not uses_previous:
        # 首镜头（以及显式声明独立的镜头）没有上一镜头可承接，只用身份素材约束。
        # 走 ref2va 而不是 fl2va：两者是独立的 62GB DiT 分区，同时加载会 OOM
        # （见 run_h3.sh 的分区说明），所以整条流水线统一在 ref2va 分区上跑。
        payload = {
            "prompt": prompt,
            "aspect_ratio": project["aspect_ratio"],
            "seconds": shot["seconds"],
            "quality": project["quality"],
            "seed": seed,
            "generate_sound": _director_sound_enabled(project),
        }
        multipart: list[tuple[str, Any]] = []
        ref_meta = _director_reference_meta(project, selected_assets, limit_images=9)
        for asset, meta in ref_meta:
            path = _director_asset_path(project["id"], asset["path"])
            multipart.append(("references", (path.name, path.read_bytes(), "application/octet-stream")))
        if not ref_meta:
            raise RuntimeError("首镜头至少需要一个启用的参考素材（人物/场景身份图）")
        payload["references_meta"] = [meta for _, meta in ref_meta]
        execution["references"] = [meta.get("label") for meta in payload["references_meta"]]
        multipart.insert(0, ("payload", (None, json.dumps(payload, ensure_ascii=False))))
        endpoint = f"{FRONTEND_INTERNAL_URL}/api/ref2va"
    else:
        previous = shots[index - 1]
        previous_filename = previous.get("filename")
        if not previous_filename:
            raise RuntimeError("上一镜头尚无成片，无法建立连续性")
        previous_path = _output_path(str(previous_filename))
        if not previous_path.is_file():
            raise RuntimeError("上一镜头视频文件不存在")
        clip_path = DIRECTOR_ASSET_ROOT / project["id"] / "continuity" / f"{previous['id']}_tail.mp4"
        # Ref2VA 参考视频时长硬下限 2 秒；早于本次修复创建的项目可能存了 1.0，
        # 直接用会让镜头 2 起全部失败，所以这里再抬一次底。
        overlap = max(float(project.get("overlap_seconds") or REF_VIDEO_MIN_SECONDS), REF_VIDEO_MIN_SECONDS)
        await asyncio.to_thread(
            _extract_tail_clip,
            previous_path,
            clip_path,
            overlap,
        )
        # 切出来的片段也要自检：源视频太短、编码异常都会让它不合规，
        # 而不合规的参考视频会连带杀死整个后端 stage。
        clip_problems = await asyncio.to_thread(_reference_video_problems, clip_path, "上一镜头尾部片段")
        if clip_problems:
            raise RuntimeError(
                "无法从上一镜头切出合规的连续性片段（" + "；".join(clip_problems)
                + "）。请把项目的「衔接参考长度」调到 2 秒以上，或把该镜头的连续性改为独立镜头。"
            )
        payload = {
            "prompt": prompt,
            "aspect_ratio": project["aspect_ratio"],
            "seconds": shot["seconds"],
            "quality": project["quality"],
            "seed": seed,
            "generate_sound": _director_sound_enabled(project),
        }
        multipart = []
        # 「上一镜头尾部」要占掉一个视频槽位，所以项目自带素材按 8 图 / 2 视频收敛，
        # 总数才不会撞上 Ref2VA 的 12 个上限。continuity=previous_video 表示只靠
        # 上一镜头承接、不再叠加身份图。
        carried = [] if shot.get("continuity") == "previous_video" else selected_assets
        ref_pairs = _director_reference_meta(project, carried, limit_images=8, limit_videos=2)
        ref_meta: list[dict[str, Any]] = []
        for asset, meta in ref_pairs:
            path = _director_asset_path(project["id"], asset["path"])
            multipart.append(("references", (path.name, path.read_bytes(), "application/octet-stream")))
            ref_meta.append(meta)
        multipart.append(("references", (clip_path.name, clip_path.read_bytes(), "video/mp4")))
        ref_meta.append({"client_id": f"continuity_{previous['id']}", "kind": "video", "label": "上一镜头尾部", "role": "上一镜头连续性", "subject": "", "description": "保持末尾姿态、运动方向、人物站位与摄影机方向", "priority": "primary", "enabled": True})
        payload["references_meta"] = ref_meta
        execution["continuity_source"] = f"上一镜头尾部（{previous['id']}）"
        execution["references"] = [meta.get("label") for meta in ref_meta]
        multipart.insert(0, ("payload", (None, json.dumps(payload, ensure_ascii=False))))
        endpoint = f"{FRONTEND_INTERNAL_URL}/api/ref2va"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(endpoint, files=multipart)
    if response.status_code != 200:
        raise RuntimeError(_error_text(response))
    result = response.json()
    job_id = str(result["id"])
    record = jobs.get(job_id, {})
    execution["job_id"] = job_id
    execution["model_prompt"] = record.get("effective_prompt") or record.get("prompt") or prompt
    execution["reference_manifest"] = record.get("reference_manifest") or []
    execution["request_snapshot"] = record.get("request_snapshot")
    return job_id


def _concat_director_project(project: dict[str, Any]) -> str:
    project_id = str(project["id"])
    shots = sorted(project["shots"], key=lambda item: item["order"])
    files = [_output_path(str(shot["filename"])) for shot in shots]
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("部分镜头文件缺失，无法合成")
    workdir = DIRECTOR_ASSET_ROOT / project_id / "output"
    workdir.mkdir(parents=True, exist_ok=True)
    normalized = workdir / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    normalized_files: list[Path] = []
    output_config = config_section("output")
    use_nvenc = str(output_config.get("video_codec", "h264")).lower() == "h264_nvenc"
    nvidia_library_dir = config_path("output", "nvidia_library_dir")
    ffmpeg_env = os.environ.copy()
    if nvidia_library_dir.is_dir():
        existing_library_path = ffmpeg_env.get("LD_LIBRARY_PATH", "")
        ffmpeg_env["LD_LIBRARY_PATH"] = (
            f"{nvidia_library_dir}:{existing_library_path}" if existing_library_path else str(nvidia_library_dir)
        )
    for index, source in enumerate(files):
        target = normalized / f"{index:03d}.mp4"
        shot = shots[index]
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
        ]
        trim_start = float(shot.get("trim_start") or 0)
        trim_end = shot.get("trim_end")
        if trim_start > 0:
            command += ["-ss", f"{trim_start:.3f}"]
        command += ["-i", str(source)]
        if trim_end is not None and float(trim_end) > trim_start:
            command += ["-t", f"{float(trim_end) - trim_start:.3f}"]
        command += [
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=24",
        ]
        if use_nvenc:
            command += [
                "-c:v", "h264_nvenc",
                "-gpu", str(output_config.get("nvenc_gpu", 0)),
                "-preset", str(output_config.get("nvenc_preset", "p1")),
                "-tune", str(output_config.get("nvenc_tune", "ll")),
                "-rc", str(output_config.get("nvenc_rate_control", "constqp")),
                "-qp", str(output_config.get("nvenc_qp", 18)),
            ]
        else:
            command += ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
        command += [
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            "-af", f"volume={float(shot.get('audio_gain') or 1):.3f}",
            "-movflags", "+faststart", str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False, env=ffmpeg_env)
        if completed.returncode != 0 and use_nvenc:
            software_command = command[:]
            codec_index = software_command.index("h264_nvenc") - 1
            audio_index = software_command.index("-c:a")
            software_command[codec_index:audio_index] = ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
            completed = subprocess.run(
                software_command, capture_output=True, text=True, timeout=600, check=False, env=ffmpeg_env
            )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"镜头 {index + 1} 统一编码失败")
        normalized_files.append(target)
    concat_file = workdir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in normalized_files),
        encoding="utf-8",
    )
    filename = f"director_{project_id[-8:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / filename
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "导演成片拼接失败")
    return filename


@app.post("/api/director/projects/{project_id}/assemble")
async def assemble_director_project(project_id: str, request: DirectorAssembleRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shots = sorted(project.get("shots", []), key=lambda item: item.get("order", 0))
    if not shots or any(shot.get("status") not in {"approved", "completed"} for shot in shots):
        raise HTTPException(status_code=409, detail="所有镜头审核通过或完成后才能导出成片")
    try:
        filename = await asyncio.to_thread(_concat_director_project, project)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=502, detail=f"成片导出失败：{exc}") from exc
    project["output_filename"] = filename
    project["status"] = "completed"
    project["message"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


async def _director_scheduler() -> None:
    while True:
        try:
            changed = False
            for project in list(director_projects.values()):
                if project.get("status") != "running":
                    continue
                review_mode = _project_review_mode(project)
                shots = sorted(project.get("shots", []), key=lambda item: item.get("order", 0))
                active = False
                for shot in shots:
                    job_id = shot.get("job_id")
                    if not job_id:
                        continue
                    record = jobs.get(str(job_id))
                    if record is None:
                        continue
                    status = str(record.get("status") or "queued")
                    if status not in TERMINAL_STATUSES:
                        shot["status"] = status
                        active = True
                    elif status == "completed" and shot.get("status") not in {"approved", "completed", "awaiting_review"}:
                        shot["status"] = "completed" if review_mode == "pipeline" else "awaiting_review"
                        shot["filename"] = record.get("filename")
                        shot["message"] = None
                        changed = True
                    elif status in {"failed", "cancelled"}:
                        shot["status"] = status
                        shot["message"] = record.get("message")
                        project["status"] = "paused"
                        changed = True
                if active or project.get("status") != "running":
                    continue
                if review_mode == "review_gate" and any(shot.get("status") == "awaiting_review" for shot in shots):
                    continue
                pending = next((shot for shot in shots if shot.get("status") in {"draft", "ready"}), None)
                if pending is not None:
                    try:
                        pending["job_id"] = await _director_submit_shot(project, pending)
                        pending["status"] = "queued"
                        pending["message"] = None
                    except Exception as exc:
                        pending["status"] = "failed"
                        pending["message"] = str(exc)
                        project["status"] = "paused"
                    project["updated_at"] = int(time.time())
                    changed = True
                elif shots and all(shot.get("status") in {"approved", "completed"} for shot in shots):
                    try:
                        project["output_filename"] = await asyncio.to_thread(_concat_director_project, project)
                        project["status"] = "completed"
                        project["message"] = None
                    except Exception as exc:
                        project["status"] = "paused"
                        project["message"] = str(exc)
                    project["updated_at"] = int(time.time())
                    changed = True
            if changed:
                async with director_lock:
                    _save_director_projects_unlocked()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(4)


@app.post("/api/director/projects/{project_id}/start")
async def start_director_project(project_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    if not project.get("shots"):
        raise HTTPException(status_code=409, detail="请先添加至少一个分镜")
    _sync_model_state_from_disk()
    if H3_PROVIDER.is_local and str(model_state.get("status")) == "loading":
        raise HTTPException(status_code=503, detail="模型正在切换，请等待加载完成后再开始导演生成")
    partition = str(model_state.get("partition") or H3_PARTITION).lower()
    if H3_PROVIDER.is_local and partition not in {"ref2va", "fl2va"}:
        raise HTTPException(status_code=409, detail="当前模型分区不支持导演模式")
    first_pending = next((shot for shot in sorted(project["shots"], key=lambda item: item["order"]) if shot.get("status") in {"draft", "ready"}), None)
    if first_pending:
        report = _director_continuity_report(project, first_pending)
        if not report["ready"]:
            raise HTTPException(status_code=409, detail="连续性检查未通过：" + "；".join(item["message"] for item in report["issues"] if item["level"] == "error"))
    project["status"] = "running"
    project["message"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/shots/{shot_id}/review")
async def review_director_shot(project_id: str, shot_id: str, request: DirectorReviewRequest) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project.get("shots", []) if item.get("id") == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    if shot.get("status") not in {"awaiting_review", "completed", "approved"}:
        raise HTTPException(status_code=409, detail="该镜头当前没有可审核的视频版本")
    shot["review_note"] = request.note.strip()
    if request.approved:
        shot["status"] = "approved"
        if project.get("status") in {"paused", "completed"} and _project_review_mode(project) == "review_gate":
            project["status"] = "running"
    else:
        shot.update(status="draft", job_id=None, filename=None, message=request.note.strip() or "审核退回")
        # A rejected shot is independently regenerated. Pipeline mode does not
        # invalidate sibling shots that are already running or completed.
        project["status"] = "running"
        project["output_filename"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/pause")
async def pause_director_project(project_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    project["status"] = "paused"
    # 暂停不仅要阻止调度下一镜，也应取消当前尚未完成的后端请求。否则界面显示
    # “已暂停”，GPU 却仍继续跑半小时，项目也因为 queued/in_progress 镜头而无法
    # 删除。已生成完的文件不受影响。
    had_active_backend_task = False
    for shot in project.get("shots", []):
        job_id = str(shot.get("job_id") or "")
        if not job_id or shot.get("status") in {"approved", "completed", "awaiting_review"}:
            continue
        record = jobs.get(job_id)
        if record is None or record.get("status") in TERMINAL_STATUSES:
            continue
        had_active_backend_task = True
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.delete(f"{H3_BASE_URL}/v1/videos/{job_id}", headers=_headers())
            if response.status_code not in {200, 404, 409}:
                raise HTTPException(status_code=502, detail=f"后端无法停止镜头任务：{_error_text(response)}")
        except (httpx.HTTPError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"后端停止镜头失败：{exc}") from exc
        watcher = watchers.pop(job_id, None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
        await _update_job(job_id, status="cancelled", message="导演项目已停止生成")
        shot.update(status="draft", job_id=None, filename=None, message="停止生成后可重新开始本镜")
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    local_config = config_section("local")
    if had_active_backend_task and H3_PROVIDER.is_local and bool(local_config.get("pause_restart_on_stuck", True)):
        # vLLM may acknowledge DELETE while the current CUDA diffusion kernel
        # keeps running. Restart the same partition so pause really means idle.
        await _set_model_state(
            partition=H3_PARTITION, target=H3_PARTITION, status="loading",
            stage="stopping", progress=3, message="停止生成后正在重启推理引擎…", error=None,
            started_at=int(time.time()),
        )
        subprocess.Popen(
            _model_switch_command(H3_PARTITION),
            cwd=str(ROOT), start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        project["message"] = "生成已停止，正在重启推理引擎以释放 GPU；模型重新加载完成后可继续。"
        async with director_lock:
            _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/director/projects/{project_id}/shots/{shot_id}/retry")
async def retry_director_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project["shots"] if item["id"] == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    shot.update(status="draft", job_id=None, filename=None, message=None)
    for later in project["shots"]:
        if later["order"] > shot["order"] and later.get("status") != "completed":
            later.update(status="draft", job_id=None, filename=None, message=None)
    project["status"] = "running"
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.delete("/api/director/projects/{project_id}")
async def delete_director_project(project_id: str) -> dict[str, Any]:
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    cancelled_jobs: list[str] = []
    # 删除是明确的破坏性操作，确认后应完成整个动作：先取消属于本项目的活动
    # H3 请求，再移除项目索引和素材目录，避免留下无法从 UI 管理的孤儿任务。
    for shot in project.get("shots", []):
        job_id = str(shot.get("job_id") or "")
        if not job_id:
            continue
        record = jobs.get(job_id)
        if record is None or record.get("status") in TERMINAL_STATUSES:
            continue
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.delete(f"{H3_BASE_URL}/v1/videos/{job_id}", headers=_headers())
            if response.status_code not in {200, 404, 409}:
                raise HTTPException(status_code=502, detail=f"后端无法取消镜头任务：{_error_text(response)}")
        except (httpx.HTTPError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"后端取消镜头失败：{exc}") from exc
        watcher = watchers.pop(job_id, None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
        await _update_job(job_id, status="cancelled", message="所属导演项目已删除")
        cancelled_jobs.append(job_id)
    async with director_lock:
        director_projects.pop(project_id, None)
        _save_director_projects_unlocked()
    shutil.rmtree(DIRECTOR_ASSET_ROOT / project_id, ignore_errors=True)
    return {"deleted": True, "id": project_id, "cancelled_jobs": cancelled_jobs}


@app.post("/api/jobs/{job_id}/resume-decode")
async def resume_job_decode(job_id: str) -> dict[str, Any]:
    """Resubmit a failed job while restoring its post-denoise latent."""
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    checkpoint_dir = LATENT_CHECKPOINT_ROOT / job_id
    missing = [rank for rank in (0, 1) if not (checkpoint_dir / f"rank{rank}.pt").is_file()]
    if missing:
        raise HTTPException(status_code=409, detail="该任务没有完整的去噪 checkpoint，无法只补解码")

    snapshot = _request_snapshot(record)
    snapshot["resume_checkpoint_key"] = job_id
    assets = _saved_assets(record)
    expected = int(record.get("reference_images") or 0)
    if len(assets) != expected:
        raise HTTPException(status_code=409, detail="任务参考图缓存不完整，无法安全恢复")

    multipart: list[tuple[str, Any]] = [
        ("payload", (None, json.dumps(snapshot, ensure_ascii=False)))
    ]
    for path in assets:
        multipart.append(("images", (path.name, path.read_bytes(), "application/octet-stream")))
    async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{FRONTEND_INTERNAL_URL}/api/generate", files=multipart)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"恢复解码入队失败：{_error_text(response)}")
    return response.json()


@app.post("/api/director/projects/{project_id}/shots/{shot_id}/resume-decode/{job_id}")
async def resume_director_shot_decode(project_id: str, shot_id: str, job_id: str) -> dict[str, Any]:
    """Resume a shot checkpoint and atomically attach the replacement job."""
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    shot = next((item for item in project.get("shots", []) if item.get("id") == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="镜头不存在")
    result = await resume_job_decode(job_id)
    shot.update(
        job_id=str(result["id"]), status="queued", filename=None,
        message="已从去噪 checkpoint 恢复解码",
    )
    project["status"] = "running"
    project["message"] = None
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return _director_project_view(project)


@app.post("/api/jobs/{job_id}/move")
async def move_job(job_id: str, request: MoveJobRequest) -> dict[str, Any]:
    """Move a waiting job and rebuild only the H3 FIFO waiting suffix."""
    owner_id = _execution_owner_id()
    waiting = [
        record
        for record in jobs.values()
        if record.get("status") not in TERMINAL_STATUSES and str(record.get("id")) != owner_id
    ]
    ids = [str(record.get("id")) for record in waiting]
    if job_id not in ids:
        raise HTTPException(status_code=409, detail="只能调整尚未开始的排队任务")
    index = ids.index(job_id)
    target = index - 1 if request.direction == "up" else index + 1
    if target < 0 or target >= len(waiting):
        return {"moved": False, "reason": "已经位于队列边界"}
    waiting[index], waiting[target] = waiting[target], waiting[index]

    saved: list[tuple[str, dict[str, Any], list[Path]]] = []
    for record in waiting:
        snapshot = _request_snapshot(record)
        assets = _saved_assets(record)
        expected = int(record.get("reference_images") or 0)
        if len(assets) != expected:
            raise HTTPException(
                status_code=409,
                detail=f"任务 {record.get('id')} 缺少原始参考图，无法安全重排",
            )
        saved.append((str(record.get("id")), snapshot, assets))

    # Remove the old waiting requests. The running owner is deliberately left
    # untouched, so reordering never wastes its current denoise work.
    async with httpx.AsyncClient(timeout=30.0) as client:
        for old_id, _, _ in saved:
            response = await client.delete(f"{H3_BASE_URL}/v1/videos/{old_id}", headers=_headers())
            if response.status_code not in {200, 404}:
                raise HTTPException(status_code=502, detail=f"后端无法移除排队任务 {old_id}: {_error_text(response)}")
            watcher = watchers.pop(old_id, None)
            if watcher is not None and not watcher.done():
                watcher.cancel()

    async with jobs_lock:
        for old_id, _, _ in saved:
            jobs.pop(old_id, None)
        _save_jobs_unlocked()

    mapping: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=90.0) as client:
        for old_id, snapshot, assets in saved:
            multipart: list[tuple[str, Any]] = [
                ("payload", (None, json.dumps(snapshot, ensure_ascii=False)))
            ]
            for path in assets:
                multipart.append(("images", (path.name, path.read_bytes(), "application/octet-stream")))
            response = await client.post(f"{FRONTEND_INTERNAL_URL}/api/generate", files=multipart)
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"重新入队失败：{_error_text(response)}")
            mapping[old_id] = str(response.json()["id"])

    return {
        "moved": True,
        "direction": request.direction,
        "old_to_new": mapping,
        "order": [mapping[old_id] for old_id, _, _ in saved],
    }


# ---------------------------------------------------------------------------
# 全局素材库
#
# 三种生成模式都要用同一批人物设定图/参考视频，原先每次都得重新上传一遍。
# 素材库把文件存一份，各模式按 id 引用：普通生成与 Ref2VA 由前端拉回成 File
# 对象放进表单，导演模式在后端直接复制进项目素材，避免大文件绕一圈浏览器。
#
# 入库即校验：视频按 Ref2VA 的全套约束验（时长/帧率/编码/容器/体积），
# 不合规的标记出来但仍允许保存 —— 用户可能只是想留着做别的用途，
# 但引用它的时候前端会拦住。
# ---------------------------------------------------------------------------
LIBRARY_LIMITS = {"image": 30 * 1024 * 1024, "video": 50 * 1024 * 1024, "audio": 15 * 1024 * 1024}
LIBRARY_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"},
    "video": {".mp4", ".mov"},
    "audio": {".wav", ".mp3"},
}
LIBRARY_DEFAULT_ROLES = {"image": "人物身份", "video": "动作参考", "audio": "环境声"}


def _library_kind_for(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    for kind, suffixes in LIBRARY_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    lowered = (content_type or "").lower()
    for kind in ("image", "video", "audio"):
        if lowered.startswith(f"{kind}/"):
            return kind
    return ""


@app.post("/api/library/items")
async def upload_library_items(files: list[UploadFile] = File(default=[])) -> dict[str, Any]:
    uploads = [item for item in files if item.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="请选择要入库的素材")
    LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    added: list[dict[str, Any]] = []
    rejected: list[str] = []
    for upload in uploads:
        name = upload.filename or "素材"
        kind = _library_kind_for(name, upload.content_type or "")
        if not kind:
            rejected.append(f"{name}：不支持的类型（图片 png/jpg/webp，视频 mp4/mov，音频 wav/mp3）")
            continue
        data = await upload.read()
        if len(data) > LIBRARY_LIMITS[kind]:
            rejected.append(f"{name}：超过 {LIBRARY_LIMITS[kind] // (1024 * 1024)}MB 上限")
            continue
        item_id = f"lib_{uuid.uuid4().hex[:12]}"
        suffix = Path(name).suffix.lower()
        stored = LIBRARY_ROOT / f"{item_id}{suffix}"
        item: dict[str, Any] = {
            "id": item_id,
            "kind": kind,
            "name": name,
            "filename": stored.name,
            "size_bytes": len(data),
            "role": LIBRARY_DEFAULT_ROLES[kind],
            "subject": "",
            "description": "",
            "priority": "support",
            "tags": [],
            "created_at": int(time.time()),
            "issues": [],
        }
        if kind == "image":
            try:
                width, height = _inspect_image(name, data)
                item["width"], item["height"] = width, height
            except HTTPException as exc:
                rejected.append(f"{name}：{exc.detail}")
                continue
        stored.write_bytes(data)
        if kind == "video":
            # 入库就把 Ref2VA 的约束验完，别等提交生成时才发现不能用。
            problems = await asyncio.to_thread(_reference_video_problems, stored, name)
            item["issues"] = problems
            try:
                meta = await asyncio.to_thread(_probe_reference_video, stored)
                item.update(width=meta["width"], height=meta["height"],
                            duration=round(meta["duration"], 3), fps=round(meta["fps"], 3),
                            has_audio=bool(meta["audio_codecs"]))
            except Exception:
                pass
        library_items[item_id] = item
        added.append(_library_view(item))
    async with library_lock:
        _save_library_unlocked()
    if not added and rejected:
        raise HTTPException(status_code=400, detail="；".join(rejected))
    return {"added": added, "rejected": rejected, "items": await list_library_items()}


@app.get("/api/library/items")
async def list_library_items() -> list[dict[str, Any]]:
    return [
        _library_view(item)
        for item in sorted(library_items.values(), key=lambda x: x.get("created_at", 0), reverse=True)
    ]


@app.get("/api/library/items/{item_id}/file")
async def library_item_file(item_id: str) -> FileResponse:
    item = library_items.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    path = _library_path(item)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="素材文件已丢失")
    return FileResponse(path, filename=str(item.get("name") or path.name))


@app.put("/api/library/items/{item_id}")
async def update_library_item(item_id: str, request: LibraryItemUpdate) -> dict[str, Any]:
    item = library_items.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    item.update(request.model_dump(mode="json"))
    async with library_lock:
        _save_library_unlocked()
    return _library_view(item)


@app.delete("/api/library/items/{item_id}")
async def delete_library_item(item_id: str) -> dict[str, Any]:
    item = library_items.pop(item_id, None)
    if item is None:
        raise HTTPException(status_code=404, detail="素材不存在")
    path = LIBRARY_ROOT / str(item.get("filename") or "")
    path.unlink(missing_ok=True)
    async with library_lock:
        _save_library_unlocked()
    return {"deleted": item_id}


@app.post("/api/director/projects/{project_id}/assets/from-library")
async def import_library_into_director(project_id: str, request: LibraryImportRequest) -> dict[str, Any]:
    """把素材库条目复制进导演项目。

    在后端直接复制而不是让浏览器下载再上传：素材可能是几十 MB 的视频，
    绕一圈客户端既慢又容易超时。
    """
    project = director_projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="导演项目不存在")
    project_root = DIRECTOR_ASSET_ROOT / project_id / "references"
    project_root.mkdir(parents=True, exist_ok=True)
    added: list[dict[str, Any]] = []
    for item_id in request.item_ids:
        item = library_items.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"素材 {item_id} 不存在")
        source = _library_path(item)
        if not source.is_file():
            raise HTTPException(status_code=404, detail=f"素材文件已丢失：{item.get('name')}")
        asset_id = f"asset_{uuid.uuid4().hex[:10]}"
        destination = project_root / f"{asset_id}{source.suffix.lower()}"
        await asyncio.to_thread(shutil.copyfile, source, destination)
        asset = {
            "id": asset_id,
            "kind": item["kind"],
            "name": item.get("name") or destination.name,
            "path": str(destination.relative_to(DIRECTOR_ASSET_ROOT / project_id)),
            "size_bytes": int(item.get("size_bytes") or destination.stat().st_size),
            "role": item.get("role") or LIBRARY_DEFAULT_ROLES.get(item["kind"], "参考"),
            "subject": item.get("subject") or "",
            "description": item.get("description") or "",
            "priority": item.get("priority") or "support",
            "enabled": True,
            "library_id": item_id,
            "created_at": int(time.time()),
        }
        project["assets"].append(asset)
        added.append(asset)
    project["updated_at"] = int(time.time())
    async with director_lock:
        _save_director_projects_unlocked()
    return {**_director_project_view(project), "added_assets": added}


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    model = _eta_model()
    active = [record for record in jobs.values() if record.get("status") not in TERMINAL_STATUSES]
    views = [_decorate_job(record, model) for record in sorted(active, key=lambda item: item.get("created_at", 0))]
    queue_index = 0
    for view in views:
        if view.get("status") == "queued":
            queue_index += 1
            view["queue_position"] = queue_index
            view["queue_label"] = "下一条" if queue_index == 1 else f"第 {queue_index + 1} 条"
    return views


@app.get("/api/jobs/history")
async def list_job_history() -> list[dict[str, Any]]:
    """Completed/failed/cancelled records live outside the active queue."""
    model = _eta_model()
    history = sorted(
        (record for record in jobs.values() if record.get("status") in TERMINAL_STATUSES),
        key=lambda item: item.get("updated_at") or item.get("created_at", 0),
        reverse=True,
    )
    return [_decorate_job(record, model) for record in history]


@app.delete("/api/jobs/history/failed")
async def clear_failed_job_history() -> dict[str, Any]:
    """One-shot purge for failed/cancelled job records and their caches."""
    removed: list[str] = []
    async with jobs_lock:
        for job_id, record in list(jobs.items()):
            if record.get("status") not in {"failed", "cancelled"}:
                continue
            removed.append(job_id)
            jobs.pop(job_id, None)
            shutil.rmtree(LATENT_CHECKPOINT_ROOT / job_id, ignore_errors=True)
            shutil.rmtree(QUEUE_ASSET_ROOT / job_id, ignore_errors=True)
        _save_jobs_unlocked()
    return {"deleted": len(removed), "ids": removed}


@app.get("/api/jobs/{job_id}/assets/{asset_index}")
async def job_asset(job_id: str, asset_index: int) -> FileResponse:
    """Serve a private cached input image so a task can refill the editor."""
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    assets = _saved_assets(record)
    if asset_index < 0 or asset_index >= len(assets):
        raise HTTPException(status_code=404, detail="任务图片不存在")
    path = assets[asset_index]
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="无效任务图片路径") from exc
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    return FileResponse(path, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, purge_file: bool = False) -> dict[str, Any]:
    """删除任务记录，并同步删掉后端存档。

    对 queued/in_progress 的任务，后端 DELETE 会先 task.cancel()，等于中止生成。
    """
    async with jobs_lock:
        record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    watcher = watchers.pop(job_id, None)
    if watcher is not None and not watcher.done():
        watcher.cancel()

    backend_result = "skipped"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.delete(f"{H3_BASE_URL}/v1/videos/{job_id}", headers=_headers())
        if response.status_code == 200:
            backend_result = "deleted"
        elif response.status_code == 404:
            backend_result = "absent"
        elif response.status_code == 409:
            # 取消尚未落地。恢复监听，让用户稍后重试，别留下半删状态。
            _start_watcher(job_id)
            raise HTTPException(status_code=409, detail="后端正在取消该任务，请稍后重试删除")
        else:
            backend_result = f"error: {_error_text(response)}"
    except (httpx.HTTPError, RuntimeError) as exc:
        backend_result = f"unreachable: {exc}"

    removed_file = None
    filename = record.get("filename")
    if purge_file and filename:
        path = _output_path(str(filename))
        if path.is_file():
            path.unlink()
            removed_file = path.name

    async with jobs_lock:
        jobs.pop(job_id, None)
        _save_jobs_unlocked()
    shutil.rmtree(LATENT_CHECKPOINT_ROOT / job_id, ignore_errors=True)
    return {"deleted": True, "id": job_id, "backend": backend_result, "removed_file": removed_file}


# 探测结果缓存。ffprobe 单次约 25ms，文件写完就不再变，按 (mtime, size) 失效足够。
_probe_cache: dict[str, tuple[float, int, dict[str, Any]]] = {}


def _probe_video(path: Path, mtime: float, size: int) -> dict[str, Any]:
    """探测 mp4 的真实画面尺寸与时长。

    前端要用它在视频加载前就锁定预览区比例 —— H3 的产物既有 1344×768 横屏也有
    768×1024 竖屏，不预先知道比例就必然出现布局跳变。
    """
    cached = _probe_cache.get(path.name)
    if cached is not None and cached[0] == mtime and cached[1] == size:
        return cached[2]

    probed: dict[str, Any] = {"width": None, "height": None, "duration_s": None}
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        payload = json.loads(completed.stdout)
        stream = next(iter(payload.get("streams") or []), {})
        width, height = stream.get("width"), stream.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            probed["width"], probed["height"] = width, height
        duration = (payload.get("format") or {}).get("duration")
        if duration is not None:
            probed["duration_s"] = round(float(duration), 3)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        # 探测失败不该让整个作品库 500。前端拿到 null 会退回读 video 元数据。
        pass

    _probe_cache[path.name] = (mtime, size, probed)
    return probed


@app.get("/api/videos")
async def list_videos() -> list[dict[str, Any]]:
    model = _eta_model()
    items: list[dict[str, Any]] = []
    candidates = list(OUTPUT_ROOT.glob("*.mp4")) + list(ROOT.glob("*.mp4"))
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        linked = next((job for job in jobs.values() if job.get("filename") == path.name), None)
        # ffprobe 是阻塞子进程，丢到线程里，别卡住事件循环上正在轮询的任务监听器
        probed = await asyncio.to_thread(_probe_video, path, stat.st_mtime, stat.st_size)
        items.append(
            {
                "filename": path.name,
                "url": f"/media/{path.name}",
                "size_bytes": stat.st_size,
                "modified_at": int(stat.st_mtime),
                "width": probed["width"],
                "height": probed["height"],
                "duration_s": probed["duration_s"],
                "job": _decorate_job(linked, model) if linked else None,
            }
        )
    return items


def _safe_media_path(filename: str) -> Path:
    # Generated videos live under the organized output directory.  Keep the
    # root-level fallback for legacy files, but resolve through the same helper
    # used by the director scheduler so /media URLs never point at a phantom
    # root-level path.
    return _safe_output_video_path(filename)


@app.delete("/api/videos/{filename}")
async def delete_video(filename: str) -> dict[str, Any]:
    path = _safe_media_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="视频不存在")
    path.unlink()
    _probe_cache.pop(path.name, None)

    # 解除任务记录上的文件关联，否则库里会留一条指向空文件的记录
    async with jobs_lock:
        for record in jobs.values():
            if record.get("filename") == path.name:
                record["filename"] = None
                record["file_removed_at"] = int(time.time())
        _save_jobs_unlocked()
    return {"deleted": True, "filename": path.name}


@app.get("/media/{filename}")
async def media(filename: str, download: bool = False) -> FileResponse:
    path = _safe_media_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="视频不存在")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        content_disposition_type="attachment" if download else "inline",
    )
