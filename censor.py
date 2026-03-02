#!/usr/bin/env python3
"""
Автоматическая цензура мата в видео/аудио с faster-whisper + ffmpeg.

Принимает любой медиафайл (определение типа через ffprobe, без привязки
к расширениям). Для видео с несколькими аудиодорожками — интерактивный
выбор в терминале. Сохраняет оригинальное качество аудио (автодетект
битрейта). Выводит зацензурированное видео + отдельные обработанные
аудиодорожки.

Особенности:
- Edge-preserve цензура: по умолчанию глушится центр слова, края остаются.
- Режим --hard: отключает edge-preserve и использует более широкий паддинг.
- Принимает файлы и директории: папки сканируются рекурсивно на медиафайлы.
- Автодетект типа медиа через ffprobe (не по расширению файла).
- Автодетект и сохранение оригинального битрейта аудио.
- Интерактивный выбор дорожек при нескольких аудиодорожках.
- Экспорт отдельных обработанных аудиодорожек в оригинальном кодеке.
- Структурная обратная связь по этапам [START]/[DONE]/[SKIP]/[ERROR].
- Continue + summary на ошибках дорожек/файлов.
- CPU fallback для Whisper, если CUDA недоступна.
- JSON-отчет по запуску (--report-json).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ленивый импорт faster-whisper (для --info без загрузки модели)
WhisperModel = None


# ============================================================================
# МЕЖПРОЦЕССНАЯ КООРДИНАЦИЯ (file-based locking)
# ============================================================================

if sys.platform == "win32":
    import msvcrt

    def _flock_exclusive(fd: Any) -> None:
        """Блокирующий лок на Windows через polling (msvcrt.LK_NBLCK).

        msvcrt.LK_LOCK делает всего ~10 попыток по 1 сек — недостаточно
        для ожидания длинных транскрипций. Используем бесконечный poll
        с LK_NBLCK + sleep, чтобы ждать столько, сколько нужно.
        """
        while True:
            fd.seek(0)
            try:
                msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(2)

    def _flock_try(fd: Any) -> bool:
        fd.seek(0)
        try:
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _flock_release(fd: Any) -> None:
        fd.seek(0)
        try:
            msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _flock_exclusive(fd: Any) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _flock_try(fd: Any) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _flock_release(fd: Any) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class ResourceSemaphore:
    """Межпроцессный семафор через N файловых локов.

    Позволяет до ``slots`` процессам одновременно держать ресурс.
    Процесс захватывает любой свободный слот; если все заняты —
    ждёт (polling) освобождения любого из них.

    Типичное использование: ограничить число параллельных процессов
    censor.py, которые одновременно держат Whisper-модель в GPU.
    """

    def __init__(
        self,
        name: str,
        slots: int = 2,
        label: str = "",
        *,
        enabled: bool = True,
    ):
        self.slots = max(1, slots)
        self.label = label or name
        self.enabled = enabled
        self._slot_paths = [
            CACHE_DIR / f".{name}_{i}.lock" for i in range(self.slots)
        ]
        self._fd: Any = None
        self._slot: int = -1

    def __enter__(self) -> "ResourceSemaphore":
        if not self.enabled:
            return self
        self._slot_paths[0].parent.mkdir(parents=True, exist_ok=True)

        # Попытка захватить любой свободный слот (non-blocking)
        for i, path in enumerate(self._slot_paths):
            fd = open(path, "w")  # noqa: SIM115
            if _flock_try(fd):
                self._fd = fd
                self._slot = i
                return self
            fd.close()

        # Все слоты заняты — ждём освобождения любого
        log(
            f"⏳ Ожидание: {self.label} "
            f"(все {self.slots} слот(а/ов) заняты другими процессами)..."
        )
        while True:
            for i, path in enumerate(self._slot_paths):
                fd = open(path, "w")  # noqa: SIM115
                if _flock_try(fd):
                    self._fd = fd
                    self._slot = i
                    log(f"🔓 {self.label} — слот {i} получен")
                    return self
                fd.close()
            time.sleep(2)

    def __exit__(self, *args: Any) -> bool:
        if self._fd is not None:
            _flock_release(self._fd)
            self._fd.close()
            self._fd = None
            self._slot = -1
        return False


# ============================================================================
# НАСТРОЙКИ
# ============================================================================

DEFAULT_MODEL = "large-v3-turbo"
KNOWN_MODELS = frozenset(
    {
        "tiny",
        "base",
        "small",
        "medium",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    }
)

SILENCE_THRESHOLD_DB = -50
BEEP_FREQ = 1000  # Гц

DEFAULT_PAD_MS = 25
HARD_PAD_MS = 50
DEFAULT_EDGE_KEEP_MS = 15
DEFAULT_MIN_CENSOR_MS = 80
MERGE_GAP_MS = 40

# Расширения файлов (для совместимости и output-хинтов)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".ts", ".m2ts", ".wmv"}

# Маппинг кодек → расширение для экспорта отдельных аудиодорожек
CODEC_EXT_MAP = {
    "aac": ".m4a", "mp3": ".mp3", "opus": ".opus", "vorbis": ".ogg",
    "flac": ".flac", "ac3": ".ac3", "eac3": ".eac3", "dts": ".dts",
    "pcm_s16le": ".wav", "pcm_f32le": ".wav", "wmav2": ".wma",
}

# Пути
SCRIPT_DIR = Path(__file__).parent
SWEARS_FILE = SCRIPT_DIR / "swears.txt"
SWEARS_FILE_ALT = Path.home() / ".config" / "censor" / "swears.txt"
CACHE_DIR = Path.home() / ".cache" / "censor"

# Глобальные кеши и временные артефакты
_duration_cache: dict[tuple[str, int, float], float] = {}
_temp_files: set[Path] = set()


# ============================================================================
# ОШИБКИ
# ============================================================================


class CensorErrorBase(Exception):
    """Базовая ошибка пайплайна."""


class DependencyError(CensorErrorBase):
    """Нет внешней зависимости (ffmpeg/ffprobe)."""


class ValidationError(CensorErrorBase):
    """Ошибка валидации аргументов/ввода."""


class ProbeError(CensorErrorBase):
    """Ошибка ffprobe."""


class ExtractError(CensorErrorBase):
    """Ошибка извлечения/конвертации аудио."""


class TranscriptionError(CensorErrorBase):
    """Ошибка транскрипции."""


class CensorApplyError(CensorErrorBase):
    """Ошибка применения цензуры."""


class EncodeError(CensorErrorBase):
    """Ошибка кодирования/копирования аудио."""


class AssembleError(CensorErrorBase):
    """Ошибка сборки финального видео."""


# ============================================================================
# ТИПЫ ДАННЫХ
# ============================================================================


@dataclass
class AudioTrack:
    stream_index: int
    audio_index: int
    codec: str
    channels: int
    sample_rate: int
    title: str
    bitrate: Optional[int] = None  # бит/с, например 192000


@dataclass
class Config:
    model_name: str = DEFAULT_MODEL
    use_beep: bool = False
    pad_ms: int = DEFAULT_PAD_MS
    edge_keep_ms: int = DEFAULT_EDGE_KEEP_MS
    edge_keep_enabled: bool = True
    min_censor_ms: int = DEFAULT_MIN_CENSOR_MS
    track_filter: Optional[list[int]] = None
    verbose: bool = False
    language: str = "ru"
    report_json_path: Optional[Path] = None
    max_gpu_slots: int = 2


@dataclass
class WordToken:
    word: str
    start: float
    end: float
    probability: Optional[float] = None


@dataclass
class SwearMatch:
    word: str
    start: float
    end: float
    probability: Optional[float] = None
    mute_start: float = 0.0
    mute_end: float = 0.0
    mode: str = "center"


@dataclass
class StageResult:
    stage: str
    ok: bool = True
    duration_ms: float = 0.0
    cache_hit: bool = False
    skipped: bool = False
    message: str = ""
    error_code: str = ""


@dataclass
class TrackResult:
    track_index: Optional[int]
    title: str
    status: str = "pending"  # ok|cached|failed|fallback_copy|copied
    found_matches: int = 0
    applied_intervals: int = 0
    censored_words: int = 0
    cache_hits: int = 0
    output_path: str = ""
    stages: list[StageResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class FileResult:
    input_path: str
    output_path: str
    media_type: str
    status: str = "pending"  # ok|partial_failed|failed
    total_matches: int = 0
    total_censored: int = 0
    duration_ms: float = 0.0
    tracks: list[TrackResult] = field(default_factory=list)
    stages: list[StageResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exported_tracks: list[str] = field(default_factory=list)


# ============================================================================
# ЛОГГИНГ И ТАЙМИНГ
# ============================================================================


def log(msg: str, prefix: str = ""):
    print(f"{prefix}{msg}")


def print_progress(current: float, total: float, prefix: str = "", width: int = 30):
    """Простой прогресс-бар в консоли."""
    if total <= 0:
        return
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r{prefix} [{bar}] {pct*100:.0f}%", end="", flush=True)
    if current >= total:
        print()


class StepTimer:
    """Контекст тайминга и статуса этапа с автоматическим логом."""

    def __init__(self, owner: Any, stage: str, prefix: str = "      "):
        self.owner = owner
        self.stage = stage
        self.prefix = prefix
        self.stage_result = StageResult(stage=stage)
        self._start = 0.0

    def __enter__(self):
        print(f"{self.prefix}[START] {self.stage}")
        self._start = time.monotonic()
        return self

    def skip(self, reason: str, cache_hit: bool = False):
        self.stage_result.message = reason
        self.stage_result.cache_hit = cache_hit
        self.stage_result.skipped = True

    def info(self, message: str):
        self.stage_result.message = message

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stage_result.duration_ms = (time.monotonic() - self._start) * 1000.0
        if exc_val is not None:
            self.stage_result.ok = False
            self.stage_result.error_code = exc_type.__name__ if exc_type else "Error"
            self.stage_result.message = str(exc_val)

        self.owner.stages.append(self.stage_result)
        if hasattr(self.owner, "cache_hits") and self.stage_result.cache_hit:
            self.owner.cache_hits += 1

        sec = self.stage_result.duration_ms / 1000.0
        if self.stage_result.ok:
            if self.stage_result.cache_hit:
                print(f"{self.prefix}[SKIP cache] {self.stage} ({sec:.2f}s)")
            elif self.stage_result.skipped:
                print(
                    f"{self.prefix}[SKIP {self.stage_result.message}] {self.stage} ({sec:.2f}s)"
                )
            else:
                print(f"{self.prefix}[DONE {sec:.2f}s] {self.stage}")
        else:
            print(
                f"{self.prefix}[ERROR {self.stage_result.error_code}] "
                f"{self.stage}: {self.stage_result.message}"
            )
        return False


def add_error(owner: Any, error: Exception | str):
    owner.errors.append(str(error))


# ============================================================================
# УТИЛИТЫ
# ============================================================================


def register_temp_file(path: Path):
    _temp_files.add(path)


def cleanup_temp_files():
    for f in list(_temp_files):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    _temp_files.clear()


MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS

_CENSOR_SUFFIX_RE = re.compile(r"_censored(_track\d+)?$")

def _is_censor_output(path: Path) -> bool:
    """Файл — результат предыдущего запуска censor.py."""
    return _CENSOR_SUFFIX_RE.search(path.stem) is not None


def collect_media_from_dir(
    directory: Path,
    *,
    recursive: bool = True,
    seen: set[Path] | None = None,
) -> list[Path]:
    """Собирает медиафайлы из директории (рекурсивно или нет).

    Возвращает отсортированный список уникальных файлов с медиа-расширениями.
    Скрытые директории (начинающиеся с точки) пропускаются.
    """
    if seen is None:
        seen = set()

    results: list[Path] = []
    iterator = directory.rglob("*") if recursive else directory.iterdir()

    for entry in sorted(iterator):
        if not entry.is_file():
            continue
        # Пропускаем скрытые файлы и файлы в скрытых директориях
        if any(part.startswith(".") for part in entry.relative_to(directory).parts):
            continue
        resolved = entry.resolve()
        if resolved in seen:
            continue
        if is_media_file(entry) and not _is_censor_output(entry):
            seen.add(resolved)
            results.append(resolved)

    return results


def run_cmd(
    cmd: list[str], capture: bool = False, quiet: bool = True
) -> subprocess.CompletedProcess:
    """Запуск команды с единым поведением."""
    try:
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True)
        if quiet:
            return subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        return subprocess.run(cmd)
    except FileNotFoundError as exc:
        raise DependencyError(f"Не найдена команда: {cmd[0]}") from exc


def run_ffmpeg_with_progress(cmd: list[str], duration: float, prefix: str = "") -> bool:
    """Запуск ffmpeg с progress=pipe."""
    if duration <= 0:
        return run_cmd(cmd, quiet=True).returncode == 0

    cmd_with_progress = cmd.copy()
    cmd_with_progress.insert(1, "-progress")
    cmd_with_progress.insert(2, "pipe:1")

    try:
        process = subprocess.Popen(
            cmd_with_progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DependencyError("Не найдена команда: ffmpeg") from exc

    current_time = 0.0
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                time_us = int(line.split("=")[1])
            except ValueError:
                continue
            current_time = time_us / 1_000_000
            print_progress(current_time, duration, prefix)
        elif line == "progress=end":
            print_progress(duration, duration, prefix)

    process.wait()
    return process.returncode == 0


def check_dependencies():
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise DependencyError(f"Отсутствуют зависимости в PATH: {', '.join(missing)}")


def check_model_name(model_name: str):
    if model_name not in KNOWN_MODELS:
        print(f"⚠️  Неизвестная модель '{model_name}'. Продолжаем попытку загрузки.")


def get_file_hash(path: Path) -> str:
    stat = path.stat()
    data = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(data.encode()).hexdigest()[:10]


def get_cache_dir(input_file: Path) -> Path:
    h = get_file_hash(input_file)
    cache = CACHE_DIR / f"{input_file.stem}_{h}"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def compute_swears_hash(swears: set[str]) -> str:
    data = "\n".join(sorted(swears)).encode("utf-8")
    return hashlib.md5(data).hexdigest()[:10]


def compute_config_signature(config: Config, swears_hash: str) -> str:
    payload = {
        "model": config.model_name,
        "beep": config.use_beep,
        "pad_ms": config.pad_ms,
        "edge_keep_ms": config.edge_keep_ms,
        "edge_keep_enabled": config.edge_keep_enabled,
        "min_censor_ms": config.min_censor_ms,
        "language": config.language,
        "swears": swears_hash,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


def get_processing_cache_dir(
    input_file: Path, config: Config, swears_hash: str
) -> Path:
    base = get_cache_dir(input_file)
    signature = compute_config_signature(config, swears_hash)
    run_dir = base / f"run_{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_duration_key(path: Path) -> tuple[str, int, float]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime


def get_audio_duration_raw(path: Path) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        return 0.0
    try:
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0.0))
    except (ValueError, KeyError, TypeError):
        return 0.0


def get_audio_duration(path: Path) -> float:
    key = get_duration_key(path)
    if key not in _duration_cache:
        _duration_cache[key] = get_audio_duration_raw(path)
    return _duration_cache[key]


def verify_output(path: Path, label: str):
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label}: пустой или отсутствующий файл: {path}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label}: ffprobe не смог прочитать файл: {path}")


def check_disk_space(path: Path, required_bytes: int):
    free = shutil.disk_usage(path.parent).free
    if free < required_bytes:
        print(
            f"⚠️  Мало места: свободно {free/1e9:.2f} GB, "
            f"ориентировочно нужно ~{required_bytes/1e9:.2f} GB"
        )


def estimate_wav_bytes(duration_s: float, sample_rate: int, channels: int) -> int:
    return int(duration_s * max(sample_rate, 1) * max(channels, 1) * 4 * 1.2)


def load_swears() -> set[str]:
    swears_file = SWEARS_FILE if SWEARS_FILE.exists() else SWEARS_FILE_ALT
    if not swears_file.exists():
        raise ValidationError(
            "Файл swears.txt не найден. " f"Пути: {SWEARS_FILE} или {SWEARS_FILE_ALT}"
        )

    swears: set[str] = set()
    for line in swears_file.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            swears.add(word)

    if not swears:
        raise ValidationError("Словарь swears.txt пуст.")

    log(f"📝 Загружено {len(swears)} слов из {swears_file.name}")
    return swears


def build_swear_matcher(swears: set[str]) -> re.Pattern[str]:
    patterns = [re.escape(w) for w in sorted(swears, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(patterns) + r")[а-яёa-z]*\b", re.IGNORECASE)


def parse_track_filter(raw: Optional[str]) -> Optional[list[int]]:
    if not raw:
        return None
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            raise ValidationError(f"Некорректный индекс дорожки: '{p}'")
        idx = int(p)
        if idx < 0:
            raise ValidationError("Индекс дорожки не может быть отрицательным")
        out.append(idx)
    if not out:
        raise ValidationError("Пустой список в --tracks")
    return sorted(set(out))


def interactive_track_selection(tracks: list[AudioTrack]) -> set[int]:
    """Интерактивный выбор аудиодорожек в терминале.

    Показывает нумерованный список с параметрами каждой дорожки (кодек,
    каналы, sample rate, битрейт). Пользователь вводит номера через запятую,
    'all', или Enter для обработки всех дорожек.
    """
    available = {t.audio_index for t in tracks}

    print("\n📊 Доступные аудиодорожки:")
    for t in tracks:
        br = f", {format_bitrate(t.bitrate)}" if t.bitrate else ""
        print(
            f"   [{t.audio_index}] {t.title} "
            f"({t.codec}, {t.channels}ch, {t.sample_rate}Hz{br})"
        )
    print("\nВведите номера дорожек через запятую (например: 0,2)")
    print("  'all' или Enter — обработать все")

    while True:
        try:
            raw = input("\n> ").strip().lower()
        except EOFError:
            return available

        if raw in ("", "all"):
            return available

        try:
            indices: set[int] = set()
            for part in raw.split(","):
                p = part.strip()
                if not p:
                    continue
                if not p.isdigit():
                    raise ValueError(f"'{p}' — не число")
                idx = int(p)
                if idx not in available:
                    raise ValueError(
                        f"дорожка {idx} не найдена "
                        f"(доступны: {sorted(available)})"
                    )
                indices.add(idx)
            if not indices:
                raise ValueError("пустой выбор")
            return indices
        except ValueError as exc:
            print(f"  ⚠️  Ошибка: {exc}. Попробуйте ещё раз.")


def export_separate_tracks(
    tracks: list[AudioTrack],
    processed_paths: dict[int, Path],
    selected: set[int],
    output_dir: Path,
    video_stem: str,
) -> list[Path]:
    """Экспортирует обработанные аудиодорожки в отдельные файлы.

    Для каждой обработанной дорожки выполняет ремукс из .mka в контейнер,
    соответствующий оригинальному кодеку (AAC → .m4a, AC3 → .ac3 и т.д.).
    Использует ffmpeg -c:a copy (без перекодирования).

    Returns:
        Список путей к экспортированным файлам.
    """
    exported: list[Path] = []
    for t in tracks:
        if t.audio_index not in selected:
            continue
        mka_path = processed_paths.get(t.audio_index)
        if not mka_path or not mka_path.exists():
            continue

        ext = CODEC_EXT_MAP.get(t.codec, ".mka")
        out_name = f"{video_stem}_censored_track{t.audio_index}{ext}"
        out_path = output_dir / out_name

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mka_path), "-c:a", "copy", str(out_path),
        ]
        ok = run_cmd(cmd, quiet=True).returncode == 0
        if ok and out_path.exists():
            exported.append(out_path)
        else:
            print(f"      ⚠️  Не удалось экспортировать дорожку {t.audio_index}")
    return exported


def get_output_path(input_path: Path, custom_output: Optional[str]) -> Path:
    if custom_output:
        return Path(custom_output).resolve()
    return input_path.parent / f"{input_path.stem}_censored{input_path.suffix}"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


# ============================================================================
# ПРОБЫ И МЕДИА
# ============================================================================


def probe_media_type(input_file: Path) -> str:
    """Определяет тип медиафайла через ffprobe.

    Анализирует потоки файла: если есть видеопоток (не обложка) — "video",
    если только аудио — "audio". Позволяет принимать любой формат файла.

    Returns:
        "video" или "audio"

    Raises:
        ValidationError: если файл не содержит ни видео, ни аудио потоков.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(input_file),
    ]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        raise ValidationError(f"ffprobe не смог прочитать файл: {input_file}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Некорректный ответ ffprobe: {input_file}") from exc

    has_video = False
    has_audio = False
    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            # Пропускаем обложки (attached pictures) — они не являются видеопотоком
            disposition = stream.get("disposition", {})
            if disposition.get("attached_pic", 0) == 1:
                continue
            codec_name = stream.get("codec_name", "")
            if codec_name in ("mjpeg", "png", "bmp"):
                continue
            has_video = True
        elif codec_type == "audio":
            has_audio = True

    if has_video:
        return "video"
    if has_audio:
        return "audio"
    raise ValidationError(f"Файл не содержит видео или аудио потоков: {input_file}")


def get_audio_tracks(input_file: Path) -> list[AudioTrack]:
    """Извлекает список аудиодорожек из видеофайла через ffprobe.

    Парсит bit_rate каждого потока для сохранения оригинального качества
    при перекодировании.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(input_file),
    ]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        raise ProbeError(f"ffprobe ошибка для {input_file}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"Некорректный JSON ffprobe для {input_file}") from exc

    tracks: list[AudioTrack] = []
    audio_idx = 0
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            raw_br = stream.get("bit_rate")
            bitrate = int(raw_br) if raw_br else None
            tracks.append(
                AudioTrack(
                    stream_index=stream.get("index", 0),
                    audio_index=audio_idx,
                    codec=stream.get("codec_name", "aac"),
                    channels=stream.get("channels", 2),
                    sample_rate=int(stream.get("sample_rate", 48000)),
                    title=stream.get("tags", {}).get("title", f"Track {audio_idx}"),
                    bitrate=bitrate,
                )
            )
            audio_idx += 1
    return tracks


def get_audio_info(input_file: Path) -> Optional[AudioTrack]:
    """Извлекает информацию об аудиофайле (кодек, каналы, sample rate, битрейт).

    Для битрейта сначала проверяет stream.bit_rate, затем format.bit_rate
    как fallback (для форматов, где битрейт указан только на уровне контейнера).
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_file),
    ]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    format_br = info.get("format", {}).get("bit_rate")
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            raw_br = stream.get("bit_rate") or format_br
            bitrate = int(raw_br) if raw_br else None
            return AudioTrack(
                stream_index=stream.get("index", 0),
                audio_index=0,
                codec=stream.get("codec_name", "mp3"),
                channels=stream.get("channels", 2),
                sample_rate=int(stream.get("sample_rate", 44100)),
                title=input_file.stem,
                bitrate=bitrate,
            )
    return None


def to_whisper_wav(
    src: Path,
    dst: Path,
    audio_index: Optional[int] = None,
    show_progress: bool = False,
    prefix: str = "",
) -> bool:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if audio_index is not None:
        cmd.extend(["-map", f"0:a:{audio_index}"])
    cmd.extend(["-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(dst)])
    if show_progress:
        duration = get_audio_duration(src)
        ok = run_ffmpeg_with_progress(cmd, duration, prefix)
    else:
        ok = run_cmd(cmd, quiet=True).returncode == 0
    if ok:
        verify_output(dst, "to_whisper_wav")
    return ok


def to_full_wav(
    src: Path,
    dst: Path,
    audio_index: Optional[int] = None,
    show_progress: bool = False,
    prefix: str = "",
) -> bool:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if audio_index is not None:
        cmd.extend(["-map", f"0:a:{audio_index}"])
    cmd.extend(["-acodec", "pcm_f32le", str(dst)])
    if show_progress:
        duration = get_audio_duration(src)
        ok = run_ffmpeg_with_progress(cmd, duration, prefix)
    else:
        ok = run_cmd(cmd, quiet=True).returncode == 0
    if ok:
        verify_output(dst, "to_full_wav")
    return ok


def copy_audio_source(src: Path, output: Path, audio_index: Optional[int]) -> bool:
    if audio_index is None:
        shutil.copy2(src, output)
        verify_output(output, "copy_audio_source")
        return True

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        f"0:a:{audio_index}",
        "-c:a",
        "copy",
        str(output),
    ]
    ok = run_cmd(cmd, quiet=True).returncode == 0
    if ok:
        verify_output(output, "copy_audio_source(track)")
    return ok


def encode_audio(
    input_wav: Path,
    output_file: Path,
    track: AudioTrack,
    show_progress: bool = False,
    prefix: str = "",
) -> bool:
    """Кодирует WAV обратно в оригинальный кодек.

    Битрейт определяется автоматически из метаданных трека (track.bitrate).
    Если оригинальный битрейт недоступен — используются разумные дефолты.
    """
    fallback_bitrate = {
        "aac": "192k", "mp3": "192k", "opus": "128k",
        "vorbis": "192k", "ac3": "384k", "eac3": "384k", "dts": "768k",
    }
    if track.bitrate and track.bitrate > 0:
        br = f"{track.bitrate // 1000}k"
    else:
        br = fallback_bitrate.get(track.codec, "192k")

    codec_map = {
        "aac": ["-c:a", "aac", "-b:a", br],
        "mp3": ["-c:a", "libmp3lame", "-b:a", br],
        "opus": ["-c:a", "libopus", "-b:a", br],
        "vorbis": ["-c:a", "libvorbis", "-b:a", br],
        "flac": ["-c:a", "flac"],
        "ac3": ["-c:a", "ac3", "-b:a", br],
        "eac3": ["-c:a", "eac3", "-b:a", br],
        "dts": ["-c:a", "dca", "-b:a", br],
    }
    codec_args = codec_map.get(track.codec, ["-c:a", "aac", "-b:a", br])

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_wav),
        "-ar",
        str(track.sample_rate),
        "-ac",
        str(track.channels),
        *codec_args,
        str(output_file),
    ]
    if show_progress:
        duration = get_audio_duration(input_wav)
        ok = run_ffmpeg_with_progress(cmd, duration, prefix)
    else:
        ok = run_cmd(cmd, quiet=True).returncode == 0
    if ok:
        verify_output(output_file, "encode_audio")
    return ok


def assemble_video(
    input_file: Path,
    audio_files: list[Path],
    output_file: Path,
    tracks: list[AudioTrack],
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        "-i",
        str(input_file),
    ]
    for audio in audio_files:
        cmd.extend(["-i", str(audio)])

    cmd.extend(["-map", "0:v?"])
    for i in range(len(audio_files)):
        cmd.extend(["-map", f"{i+1}:a:0"])
    cmd.extend(["-map", "0:s?"])
    cmd.extend(["-c:v", "copy", "-c:s", "copy", "-c:a", "copy"])

    for i, track in enumerate(tracks):
        if track.title:
            cmd.extend([f"-metadata:s:a:{i}", f"title={track.title}"])

    cmd.append(str(output_file))
    ok = run_cmd(cmd, quiet=False).returncode == 0
    if ok:
        verify_output(output_file, "assemble_video")
    return ok


def is_silent(wav_path: Path) -> bool:
    cmd = ["ffmpeg", "-i", str(wav_path), "-af", "volumedetect", "-f", "null", "-"]
    result = run_cmd(cmd, capture=True)
    for line in result.stderr.split("\n"):
        if "max_volume" in line:
            try:
                max_vol = float(line.split("max_volume:")[1].split("dB")[0].strip())
                return max_vol < SILENCE_THRESHOLD_DB
            except (ValueError, IndexError):
                pass
    return False


# ============================================================================
# WHISPER
# ============================================================================


def resolve_hf_token() -> Optional[str]:
    for env_name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        token = (os.getenv(env_name) or "").strip()
        if token:
            return token

    try:
        from huggingface_hub import get_token as hf_get_token

        token = (hf_get_token() or "").strip()
        if token:
            return token
    except Exception:
        pass

    token_path_env = os.getenv("HF_TOKEN_PATH")
    token_path = (
        Path(token_path_env)
        if token_path_env
        else Path.home() / ".cache" / "huggingface" / "token"
    )
    try:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    except Exception:
        pass
    return None


def load_whisper_model(model_name: str):
    global WhisperModel
    if WhisperModel is None:
        from faster_whisper import WhisperModel as WM

        WhisperModel = WM

    models_dir = CACHE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    local_model_dirs = list(models_dir.glob(f"models--*--faster-whisper-{model_name}"))
    local_files_only = any(p.is_dir() for p in local_model_dirs)
    hf_token = resolve_hf_token()
    # На Windows без Developer Mode symlink-кеш HF недоступен; отключаем шумный warning.
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if hf_token:
        # Дублируем в env, чтобы токен виделся внутри вызовов huggingface_hub.
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        log("🔐 HF токен найден: запросы к HF Hub будут аутентифицированы.")
    else:
        log("⚠️  HF токен не найден: загрузка с HF Hub может упереться в лимиты.")

    attempts = [("cuda", "float16"), ("cpu", "int8")]
    last_error: Optional[Exception] = None
    for device, compute_type in attempts:
        try:
            if device == "cuda":
                log(f"🤖 Загрузка модели {model_name} (CUDA {compute_type})...")
            else:
                log(f"⚠️  CUDA недоступна. Фолбек на CPU ({compute_type})...")

            return WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(models_dir),
                local_files_only=local_files_only,
                use_auth_token=hf_token,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if device == "cuda":
                continue
            break

    raise TranscriptionError(f"Не удалось загрузить модель {model_name}: {last_error}")


def transcribe(
    model,
    audio_path: Path,
    language: str = "ru",
    show_progress: bool = True,
    prefix: str = "      ",
) -> list[WordToken]:
    duration = get_audio_duration(audio_path) if show_progress else 0.0
    try:
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(str(exc)) from exc

    words: list[WordToken] = []
    for segment in segments:
        if show_progress and duration > 0:
            print_progress(segment.end, duration, f"{prefix}🎤 Транскрипция")
        if segment.words:
            for word in segment.words:
                words.append(
                    WordToken(
                        word=word.word.strip(),
                        start=float(word.start),
                        end=float(word.end),
                        probability=getattr(word, "probability", None),
                    )
                )
    return words


# ============================================================================
# ПОИСК МАТА И ИНТЕРВАЛЫ
# ============================================================================


def find_swears(
    words: list[WordToken], matcher: re.Pattern[str], config: Config
) -> list[SwearMatch]:
    matches: list[SwearMatch] = []
    for token in words:
        clean_word = re.sub(r"[^\w]", "", token.word.lower())
        if matcher.search(clean_word):
            match = SwearMatch(
                word=token.word,
                start=token.start,
                end=token.end,
                probability=token.probability,
            )
            matches.append(match)
            if config.verbose:
                conf = (
                    "n/a" if token.probability is None else f"{token.probability:.2f}"
                )
                log(
                    f"        SWEAR: '{token.word}' at {token.start:.2f}-{token.end:.2f}s "
                    f"(conf={conf})"
                )
    return matches


def merge_intervals(
    intervals: list[tuple[float, float]], merge_gap_ms: int = MERGE_GAP_MS
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    gap = merge_gap_ms / 1000.0
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_censor_intervals(
    matches: list[SwearMatch],
    config: Config,
    audio_duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    pad = config.pad_ms / 1000.0
    edge = config.edge_keep_ms / 1000.0
    min_len = config.min_censor_ms / 1000.0

    intervals: list[tuple[float, float]] = []
    for m in matches:
        base_start = max(0.0, m.start - pad)
        base_end = m.end + pad
        if audio_duration is not None:
            base_end = min(base_end, audio_duration)

        if config.edge_keep_enabled:
            mute_start = base_start + edge
            mute_end = base_end - edge
            if (mute_end - mute_start) < min_len:
                mute_start, mute_end = base_start, base_end
                m.mode = "full"
            else:
                m.mode = "center"
        else:
            mute_start, mute_end = base_start, base_end
            m.mode = "full"

        m.mute_start = max(0.0, mute_start)
        m.mute_end = max(m.mute_start, mute_end)
        if m.mute_end > m.mute_start:
            intervals.append((m.mute_start, m.mute_end))

    return merge_intervals(intervals, merge_gap_ms=MERGE_GAP_MS)


# ============================================================================
# ЦЕНЗУРА АУДИО
# ============================================================================


def generate_beep(duration_sec: float, sample_rate: int, channels: int) -> np.ndarray:
    t = np.linspace(0, duration_sec, round(sample_rate * duration_sec), dtype=np.float32)
    beep = 0.5 * np.sin(2 * np.pi * BEEP_FREQ * t)

    fade_samples = min(int(0.01 * sample_rate), len(beep) // 4)
    if fade_samples > 0:
        beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
        beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    if channels > 1:
        beep = np.tile(beep.reshape(-1, 1), (1, channels))
    return beep


def build_silence_filter_script(intervals: list[tuple[float, float]]) -> str:
    lines: list[str] = []
    for i, (start, end) in enumerate(intervals):
        in_label = "[0:a]" if i == 0 else f"[a{i}]"
        out_label = "[outa]" if i == len(intervals) - 1 else f"[a{i+1}]"
        lines.append(
            f"{in_label}volume=enable='between(t,{start:.6f},{end:.6f})':volume=0{out_label}"
        )
    return ";\n".join(lines)


def censor_silence_ffmpeg(
    input_wav: Path,
    output_wav: Path,
    intervals: list[tuple[float, float]],
    show_progress: bool = False,
    prefix: str = "      ",
) -> int:
    if not intervals:
        shutil.copy2(input_wav, output_wav)
        verify_output(output_wav, "censor_silence_ffmpeg(copy)")
        return 0

    fd, script_path_raw = tempfile.mkstemp(
        suffix=".ffscript",
        dir=str(output_wav.parent),
        prefix="censor_",
        text=True,
    )
    os.close(fd)
    script_path = Path(script_path_raw)
    register_temp_file(script_path)
    script_path.write_text(build_silence_filter_script(intervals), encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_wav),
        "-filter_complex_script",
        str(script_path),
        "-map",
        "[outa]",
        "-acodec",
        "pcm_f32le",
        str(output_wav),
    ]
    if show_progress:
        ok = run_ffmpeg_with_progress(
            cmd, get_audio_duration(input_wav), f"{prefix}🔇 Цензура"
        )
    else:
        ok = run_cmd(cmd, quiet=True).returncode == 0
    if not ok:
        raise CensorApplyError("Ошибка ffmpeg фильтра при mute-цензуре")
    verify_output(output_wav, "censor_silence_ffmpeg")
    return len(intervals)


def censor_beep_blockwise(
    input_wav: Path,
    output_wav: Path,
    intervals: list[tuple[float, float]],
    show_progress: bool = True,
    prefix: str = "      ",
) -> int:
    import soundfile as sf

    if not intervals:
        shutil.copy2(input_wav, output_wav)
        verify_output(output_wav, "censor_beep_blockwise(copy)")
        return 0

    info = sf.info(str(input_wav))
    sr = info.samplerate
    channels = info.channels
    total_frames = info.frames

    sample_intervals = [
        (max(0, int(s * sr)), max(0, int(e * sr))) for s, e in intervals
    ]
    sample_intervals = [(s, e) for s, e in sample_intervals if e > s]

    try:
        with sf.SoundFile(
            str(output_wav),
            mode="w",
            samplerate=sr,
            channels=channels,
            subtype="FLOAT",
            format="WAV",
        ) as out_f:
            cursor = 0
            interval_idx = 0
            for block in sf.blocks(
                str(input_wav), blocksize=65536, dtype="float32", always_2d=True
            ):
                block_len = len(block)
                block_start = cursor
                block_end = block_start + block_len

                while (
                    interval_idx < len(sample_intervals)
                    and sample_intervals[interval_idx][1] <= block_start
                ):
                    interval_idx += 1

                j = interval_idx
                while j < len(sample_intervals) and sample_intervals[j][0] < block_end:
                    s, e = sample_intervals[j]
                    local_s = max(0, s - block_start)
                    local_e = min(block_len, e - block_start)
                    if local_e > local_s:
                        dur = (local_e - local_s) / sr
                        beep = generate_beep(dur, sr, channels)
                        if beep.ndim == 1:
                            beep = beep.reshape(-1, 1)
                        target_len = local_e - local_s
                        if beep.shape[0] > target_len:
                            beep = beep[:target_len]
                        elif beep.shape[0] < target_len:
                            beep = np.pad(beep, ((0, target_len - beep.shape[0]), (0, 0)), mode="constant")
                        block[local_s:local_e, :] = beep
                    j += 1

                out_f.write(block)
                cursor = block_end
                if show_progress and total_frames > 0:
                    print_progress(cursor, total_frames, f"{prefix}🔇 Цензура")
    except Exception as exc:  # noqa: BLE001
        raise CensorApplyError(f"Ошибка blockwise beep-цензуры: {exc}") from exc

    verify_output(output_wav, "censor_beep_blockwise")
    return len(sample_intervals)


def censor_audio(
    input_wav: Path,
    output_wav: Path,
    intervals: list[tuple[float, float]],
    use_beep: bool,
    show_progress: bool = True,
    prefix: str = "      ",
) -> int:
    if use_beep:
        return censor_beep_blockwise(
            input_wav=input_wav,
            output_wav=output_wav,
            intervals=intervals,
            show_progress=show_progress,
            prefix=prefix,
        )
    return censor_silence_ffmpeg(
        input_wav=input_wav,
        output_wav=output_wav,
        intervals=intervals,
        show_progress=show_progress,
        prefix=prefix,
    )


# ============================================================================
# PIPELINE ОДНОГО АУДИО-СТРИМА
# ============================================================================


def load_transcript_cache(path: Path) -> Optional[list[WordToken]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            WordToken(
                word=str(item["word"]),
                start=float(item["start"]),
                end=float(item["end"]),
                probability=item.get("probability"),
            )
            for item in raw
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"⚠️  Поврежден кеш транскрипта, пересоздаю: {path.name}")
        path.unlink(missing_ok=True)
        return None


def save_transcript_cache(path: Path, words: list[WordToken]):
    path.write_text(
        json.dumps([asdict(w) for w in words], ensure_ascii=False), encoding="utf-8"
    )


def build_track_paths(
    cache_dir: Path, audio_index: Optional[int], final_ext: str
) -> dict[str, Path]:
    prefix = "audio" if audio_index is None else f"track_{audio_index}"
    return {
        "wav_whisper": cache_dir / f"{prefix}_16k.wav",
        "wav_full": cache_dir / f"{prefix}_full.wav",
        "wav_censored": cache_dir / f"{prefix}_censored.wav",
        "transcript": cache_dir / f"{prefix}_transcript.json",
        "skip_marker": cache_dir / f"{prefix}_skip",
        "final": cache_dir / f"{prefix}_final{final_ext}",
    }


def process_audio_stream(
    model,
    source_file: Path,
    audio_index: Optional[int],
    track_info: AudioTrack,
    cache_dir: Path,
    matcher: re.Pattern[str],
    config: Config,
    final_ext: str,
    preextracted_whisper: Optional[Path] = None,
) -> TrackResult:
    title = (
        track_info.title
        if track_info
        else (source_file.stem if audio_index is None else f"Track {audio_index}")
    )
    result = TrackResult(track_index=audio_index, title=title)
    paths = build_track_paths(cache_dir, audio_index, final_ext=final_ext)
    if preextracted_whisper is not None:
        paths["wav_whisper"] = preextracted_whisper
    result.output_path = str(paths["final"])

    try:
        with StepTimer(result, "cache-check") as st:
            if paths["final"].exists():
                st.skip("cache", cache_hit=True)
                result.status = "cached"
                result.censored_words = -1
                result.found_matches = -1
                result.applied_intervals = -1
                return result

        with StepTimer(result, "extract-whisper") as st:
            if paths["wav_whisper"].exists():
                st.skip("cache", cache_hit=True)
            else:
                ok = to_whisper_wav(
                    source_file,
                    paths["wav_whisper"],
                    audio_index=audio_index,
                    show_progress=False,
                )
                if not ok:
                    raise ExtractError("Не удалось извлечь WAV для Whisper")

        silent = False
        with StepTimer(result, "silence-check") as st:
            if paths["skip_marker"].exists():
                st.skip("silent-cache", cache_hit=True)
                silent = True
            elif is_silent(paths["wav_whisper"]):
                paths["skip_marker"].touch()
                st.skip("silent")
                silent = True

        if silent:
            with StepTimer(result, "copy-silent-original"):
                if not copy_audio_source(source_file, paths["final"], audio_index):
                    raise EncodeError("Не удалось скопировать тихий исходный аудиопоток")
            result.status = "ok"
            return result

        with StepTimer(result, "transcribe") as st:
            words = load_transcript_cache(paths["transcript"])
            if words is not None:
                st.skip("cache transcript", cache_hit=True)
            else:
                words = transcribe(
                    model,
                    paths["wav_whisper"],
                    language=config.language,
                    show_progress=True,
                )
                save_transcript_cache(paths["transcript"], words)

        with StepTimer(result, "match") as st:
            matches = find_swears(words, matcher, config)
            duration = get_audio_duration(paths["wav_whisper"])
            intervals = build_censor_intervals(matches, config, audio_duration=duration)
            result.found_matches = len(matches)
            result.applied_intervals = len(intervals)
            result.censored_words = len(matches)
            if config.verbose:
                for m in matches:
                    conf = "n/a" if m.probability is None else f"{m.probability:.2f}"
                    print(
                        f"        -> {m.mode.upper()} mute {m.mute_start:.2f}-{m.mute_end:.2f}s "
                        f"for '{m.word}' (conf={conf})"
                    )
            if not intervals:
                st.skip("no-matches")

        if not intervals:
            with StepTimer(result, "copy-original"):
                if not copy_audio_source(source_file, paths["final"], audio_index):
                    raise EncodeError("Не удалось скопировать исходный аудиопоток")
            result.status = "ok"
            return result

        with StepTimer(result, "extract-full") as st:
            if paths["wav_full"].exists():
                st.skip("cache", cache_hit=True)
            else:
                dur = get_audio_duration(source_file)
                required = estimate_wav_bytes(
                    dur, track_info.sample_rate, track_info.channels
                )
                check_disk_space(paths["wav_full"], required)
                ok = to_full_wav(
                    source_file,
                    paths["wav_full"],
                    audio_index=audio_index,
                    show_progress=False,
                )
                if not ok:
                    raise ExtractError("Не удалось извлечь WAV в полном качестве")

        with StepTimer(result, "censor"):
            applied = censor_audio(
                input_wav=paths["wav_full"],
                output_wav=paths["wav_censored"],
                intervals=intervals,
                use_beep=config.use_beep,
                show_progress=True,
            )
            result.applied_intervals = applied

        with StepTimer(result, "encode"):
            ok = encode_audio(
                paths["wav_censored"], paths["final"], track_info, show_progress=False
            )
            if not ok:
                raise EncodeError("Не удалось закодировать финальное аудио")

        result.status = "ok"
        return result

    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        add_error(result, f"{type(exc).__name__}: {exc}")
        return result


# ============================================================================
# ОБРАБОТКА ФАЙЛОВ
# ============================================================================


def preextract_whisper_tracks(
    input_file: Path, tracks: list[AudioTrack], cache_dir: Path
):
    if not tracks:
        return
    max_workers = min(2, len(tracks))
    if max_workers <= 1:
        for t in tracks:
            wav = cache_dir / f"track_{t.audio_index}_16k.wav"
            if not wav.exists():
                to_whisper_wav(
                    input_file, wav, audio_index=t.audio_index, show_progress=False
                )
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for t in tracks:
            wav = cache_dir / f"track_{t.audio_index}_16k.wav"
            if wav.exists():
                continue
            fut = ex.submit(to_whisper_wav, input_file, wav, t.audio_index, False, "")
            futures[fut] = (t.audio_index, wav)

        for fut in as_completed(futures):
            idx, wav = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"      ⚠️  Предизвлечение дорожки {idx} провалилось: {exc}")
                continue
            if not ok:
                print(f"      ⚠️  Предизвлечение дорожки {idx} провалилось")
            elif not wav.exists():
                print(f"      ⚠️  Предизвлечение дорожки {idx}: файл не создан")


def process_audio_file(
    input_file: Path,
    output_file: Path,
    model,
    matcher: re.Pattern[str],
    swears_hash: str,
    config: Config,
) -> FileResult:
    """Обрабатывает аудиофайл: транскрипция → поиск мата → цензура.

    Сохраняет оригинальное качество (битрейт определяется из метаданных).
    """
    file_result = FileResult(
        input_path=str(input_file),
        output_path=str(output_file),
        media_type="audio",
    )
    file_start = time.monotonic()

    print(f"\n🎵 Вход:  {input_file}")
    print(f"📁 Выход: {output_file}")

    try:
        with StepTimer(file_result, "probe-audio", prefix="   "):
            audio_info = get_audio_info(input_file)
            if not audio_info:
                raise ProbeError("Не удалось прочитать аудиофайл")

        cache_dir = get_processing_cache_dir(input_file, config, swears_hash)
        log(f"💾 Кеш: {cache_dir}")
        print(
            f"📊 {audio_info.codec}, {audio_info.channels}ch, {audio_info.sample_rate}Hz"
        )

        track_result = process_audio_stream(
            model=model,
            source_file=input_file,
            audio_index=None,
            track_info=audio_info,
            cache_dir=cache_dir,
            matcher=matcher,
            config=config,
            final_ext=output_file.suffix,
        )
        file_result.tracks.append(track_result)
        if track_result.status == "failed":
            raise CensorErrorBase("; ".join(track_result.errors))

        final_cache_path = Path(track_result.output_path)
        with StepTimer(file_result, "finalize-output", prefix="   "):
            shutil.copy2(final_cache_path, output_file)
            verify_output(output_file, "final output audio")

        file_result.total_matches = max(0, track_result.found_matches)
        file_result.total_censored = max(0, track_result.censored_words)
        file_result.status = "ok"

    except Exception as exc:  # noqa: BLE001
        file_result.status = "failed"
        add_error(file_result, f"{type(exc).__name__}: {exc}")

    file_result.duration_ms = (time.monotonic() - file_start) * 1000.0
    if file_result.status == "ok":
        size_mb = (
            output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0.0
        )
        print(f"✅ Готово: {output_file} ({size_mb:.1f} MB)")
        print(
            f"   Найдено: {file_result.total_matches}, зацензурено: {file_result.total_censored}"
        )
    else:
        print(f"❌ Ошибка аудиофайла: {input_file.name}")
        for e in file_result.errors:
            print(f"   - {e}")
    return file_result


def process_video_file(
    input_file: Path,
    output_file: Path,
    model,
    matcher: re.Pattern[str],
    swears_hash: str,
    config: Config,
) -> FileResult:
    """Обрабатывает видеофайл: извлечение дорожек → цензура → сборка.

    Если дорожек >1 и -t не указан — интерактивный выбор в терминале.
    После сборки видео экспортирует обработанные дорожки отдельными файлами.
    Вывод: «Было» (оригинал), «Стало» (видео), «Стало 2» (отдельные дорожки).
    """
    file_result = FileResult(
        input_path=str(input_file),
        output_path=str(output_file),
        media_type="video",
    )
    file_start = time.monotonic()
    had_track_error = False
    exported: list[Path] = []

    print(f"\n🎬 Вход:  {input_file}")
    print(f"📁 Выход: {output_file}")

    try:
        with StepTimer(file_result, "probe-video", prefix="   "):
            tracks = get_audio_tracks(input_file)
            if not tracks:
                raise ProbeError("Аудиодорожки не найдены")

        available = {t.audio_index for t in tracks}
        if config.track_filter is not None:
            missing = [x for x in config.track_filter if x not in available]
            if missing:
                raise ValidationError(f"Нет дорожек с индексами: {missing}")
            selected = set(config.track_filter)
        elif len(tracks) > 1:
            selected = interactive_track_selection(tracks)
        else:
            selected = available

        # interactive_track_selection уже показывает список дорожек,
        # поэтому выводим только когда выбор был неинтерактивным
        if config.track_filter is not None or len(tracks) == 1:
            print(f"\n📊 Аудиодорожек: {len(tracks)}")
            for t in tracks:
                br = f", {format_bitrate(t.bitrate)}" if t.bitrate else ""
                mark = "✓" if t.audio_index in selected else " "
                print(
                    f"   [{mark}] {t.audio_index}: {t.title} "
                    f"({t.codec}, {t.channels}ch, {t.sample_rate}Hz{br})"
                )
        print(f"\n⚙️  Обработка дорожек: {sorted(selected)}")

        cache_dir = get_processing_cache_dir(input_file, config, swears_hash)
        log(f"💾 Кеш: {cache_dir}")

        with StepTimer(file_result, "preextract-whisper", prefix="   ") as st:
            selected_tracks = [t for t in tracks if t.audio_index in selected]
            preextract_whisper_tracks(input_file, selected_tracks, cache_dir)
            st.info(f"tracks={len(selected_tracks)}")

        audio_files_for_mux: list[Path] = []
        processed_paths: dict[int, Path] = {}
        for i, track in enumerate(tracks, 1):
            print(f"\n{'─'*56}")
            print(f"🎵 [{i}/{len(tracks)}] Дорожка {track.audio_index}: {track.title}")
            final_track_path = cache_dir / f"track_{track.audio_index}_final.mka"

            if track.audio_index not in selected:
                tr = TrackResult(
                    track_index=track.audio_index,
                    title=track.title,
                    output_path=str(final_track_path),
                )
                with StepTimer(tr, "copy-unselected") as st:
                    if final_track_path.exists():
                        tr.status = "cached"
                        st.skip("cache", cache_hit=True)
                    else:
                        ok = copy_audio_source(
                            input_file, final_track_path, track.audio_index
                        )
                        if not ok:
                            raise EncodeError(
                                "Не удалось скопировать невыбранную дорожку"
                            )
                        tr.status = "copied"
                file_result.tracks.append(tr)
                audio_files_for_mux.append(final_track_path)
                continue

            tr = process_audio_stream(
                model=model,
                source_file=input_file,
                audio_index=track.audio_index,
                track_info=track,
                cache_dir=cache_dir,
                matcher=matcher,
                config=config,
                final_ext=".mka",
                preextracted_whisper=cache_dir / f"track_{track.audio_index}_16k.wav",
            )

            if tr.status == "failed":
                had_track_error = True
                print("      ⚠️  Ошибка обработки дорожки, fallback на исходную копию")
                add_error(tr, "fallback_to_original")
                with StepTimer(tr, "fallback-copy"):
                    ok = copy_audio_source(
                        input_file, final_track_path, track.audio_index
                    )
                    if not ok:
                        raise EncodeError(
                            f"Fallback copy не удался для дорожки {track.audio_index}"
                        )
                tr.status = "fallback_copy"
                tr.output_path = str(final_track_path)

            file_result.total_matches += max(0, tr.found_matches)
            file_result.total_censored += max(0, tr.censored_words)
            file_result.tracks.append(tr)
            track_output = Path(tr.output_path)
            audio_files_for_mux.append(track_output)
            if track.audio_index in selected and tr.status != "fallback_copy":
                processed_paths[track.audio_index] = track_output

        with StepTimer(file_result, "assemble-video", prefix="   "):
            if not assemble_video(input_file, audio_files_for_mux, output_file, tracks):
                raise AssembleError("Ошибка сборки финального видео")

        # Экспорт отдельных обработанных аудиодорожек
        exported: list[Path] = []
        if processed_paths:
            with StepTimer(file_result, "export-tracks", prefix="   "):
                exported = export_separate_tracks(
                    tracks, processed_paths, selected,
                    output_file.parent, input_file.stem,
                )
                file_result.exported_tracks = [str(p) for p in exported]

        file_result.status = "partial_failed" if had_track_error else "ok"

    except Exception as exc:  # noqa: BLE001
        file_result.status = "failed"
        add_error(file_result, f"{type(exc).__name__}: {exc}")

    file_result.duration_ms = (time.monotonic() - file_start) * 1000.0
    if file_result.status in {"ok", "partial_failed"}:
        size_mb = (
            output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0.0
        )
        print(f"\n{'═'*56}")
        print(f"  Было:    {input_file}")
        print(f"  Стало:   {output_file} ({size_mb:.1f} MB)")
        if exported:
            print("  Стало 2: Отдельные дорожки:")
            for ep in exported:
                sz = ep.stat().st_size / (1024 * 1024)
                print(f"           {ep.name} ({sz:.1f} MB)")
        print(f"{'═'*56}")
        print(
            f"   Найдено: {file_result.total_matches}, "
            f"зацензурено: {file_result.total_censored}, "
            f"время: {file_result.duration_ms / 1000:.1f}s"
        )
    else:
        print(f"\n❌ Ошибка видеофайла: {input_file.name}")
        for e in file_result.errors:
            print(f"   - {e}")
    return file_result


# ============================================================================
# ОТЧЕТ
# ============================================================================


def build_run_report(file_results: list[FileResult]) -> dict[str, Any]:
    totals = {
        "files_total": len(file_results),
        "files_ok": sum(1 for r in file_results if r.status == "ok"),
        "files_partial_failed": sum(
            1 for r in file_results if r.status == "partial_failed"
        ),
        "files_failed": sum(1 for r in file_results if r.status == "failed"),
        "matches_total": sum(r.total_matches for r in file_results),
        "censored_total": sum(r.total_censored for r in file_results),
        "duration_ms_total": sum(r.duration_ms for r in file_results),
    }
    files = [jsonable(asdict(r)) for r in file_results]
    return {"totals": totals, "files": files}


def write_report_json(path: Path, report: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧾 JSON-отчет: {path}")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Быстрая цензура мата в видео и аудио (GPU/CPU fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s video.mkv
  %(prog)s podcast.mp3
  %(prog)s video.mkv --beep
  %(prog)s video.mkv --hard
  %(prog)s video.mkv --pad-ms 20 --edge-keep-ms 15
  %(prog)s video.mkv -t 0,2
  %(prog)s video.mkv --report-json report.json
  %(prog)s video.mkv --info
  %(prog)s --clear-cache

Директории:
  %(prog)s ./videos/              # рекурсивно все медиа внутри
  %(prog)s ./videos/ --no-recurse # только файлы в корне папки
  %(prog)s ./raw/ ./extra.mkv     # папка + отдельный файл
  %(prog)s ./dir1/ ./dir2/ --beep # несколько папок с параметрами
        """,
    )

    parser.add_argument("input", nargs="*", help="Входные файлы или директории с медиа")
    parser.add_argument(
        "-o", "--output", help="Выходной файл (только для одного входного)"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Модель Whisper (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("-t", "--tracks", help="Дорожки для видео, например: 0,2,4")
    parser.add_argument("--beep", action="store_true", help="Бип вместо тишины")
    parser.add_argument("--info", action="store_true", help="Показать дорожки")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Очистить кеш (без моделей)"
    )
    parser.add_argument(
        "--clear-models", action="store_true", help="Очистить скачанные модели"
    )

    parser.add_argument("--pad-ms", type=int, default=None, help="Паддинг цензуры в мс")
    parser.add_argument(
        "--edge-keep-ms",
        type=int,
        default=DEFAULT_EDGE_KEEP_MS,
        help="Оставлять края слова (мс)",
    )
    parser.add_argument(
        "--min-censor-ms",
        type=int,
        default=DEFAULT_MIN_CENSOR_MS,
        help="Минимальная длина mute-интервала (иначе глушим весь базовый интервал)",
    )
    parser.add_argument(
        "--no-edge-keep", action="store_true", help="Отключить сохранение краев"
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Алиас старого режима: no-edge-keep + pad=50ms",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Не рекурсировать в поддиректории (только файлы в корне папки)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Подробный вывод по этапам/матчам"
    )
    parser.add_argument("--report-json", help="Сохранить JSON-отчет в файл")
    parser.add_argument(
        "--language", default="ru", help="Язык транскрипции (default: ru)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Макс. параллельных процессов с GPU/моделью (0 = без лимита, default: 2)",
    )
    return parser


def handle_cache_clear():
    if CACHE_DIR.exists():
        models_dir = CACHE_DIR / "models"
        for item in CACHE_DIR.iterdir():
            if item == models_dir:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(f"🗑️  Кеш очищен (модели сохранены): {CACHE_DIR}")
    else:
        print("Кеш пуст")


def handle_models_clear():
    models_dir = CACHE_DIR / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
        print(f"🗑️  Модели удалены: {models_dir}")
    else:
        print("Модели не найдены")


def build_config(args: argparse.Namespace, track_filter: Optional[list[int]]) -> Config:
    if args.hard:
        pad_ms = HARD_PAD_MS if args.pad_ms is None else args.pad_ms
        edge_keep_enabled = False
    else:
        pad_ms = DEFAULT_PAD_MS if args.pad_ms is None else args.pad_ms
        edge_keep_enabled = not args.no_edge_keep

    cfg = Config(
        model_name=args.model,
        use_beep=args.beep,
        pad_ms=pad_ms,
        edge_keep_ms=args.edge_keep_ms,
        edge_keep_enabled=edge_keep_enabled,
        min_censor_ms=args.min_censor_ms,
        track_filter=track_filter,
        verbose=args.verbose,
        language=args.language,
        report_json_path=Path(args.report_json).resolve() if args.report_json else None,
        max_gpu_slots=args.max_parallel,
    )

    if cfg.pad_ms < 0 or cfg.edge_keep_ms < 0 or cfg.min_censor_ms < 0:
        raise ValidationError("Параметры миллисекунд не могут быть отрицательными")
    return cfg


def format_bitrate(bitrate: Optional[int]) -> str:
    """Форматирует битрейт для вывода (например 192000 → '192 kbps')."""
    if not bitrate:
        return ""
    return f"{bitrate // 1000} kbps"


def print_info(input_files: list[Path]):
    for input_path in input_files:
        print(f"\n📁 {input_path}")
        try:
            media_type = probe_media_type(input_path)
        except ValidationError as exc:
            print(f"    ❌ {exc}")
            continue

        if media_type == "audio":
            info = get_audio_info(input_path)
            if info:
                br = f", {format_bitrate(info.bitrate)}" if info.bitrate else ""
                print(
                    f"    🎵 Аудио: {info.codec}, {info.channels}ch, "
                    f"{info.sample_rate}Hz{br}"
                )
            else:
                print("    ❌ Не удалось прочитать аудио")
        else:
            tracks = get_audio_tracks(input_path)
            if not tracks:
                print("    ❌ Аудиодорожки не найдены")
            for t in tracks:
                br = f", {format_bitrate(t.bitrate)}" if t.bitrate else ""
                print(f"    [{t.audio_index}] {t.title}")
                print(f"        {t.codec}, {t.channels}ch, {t.sample_rate}Hz{br}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.clear_cache:
            handle_cache_clear()
            return
        if args.clear_models:
            handle_models_clear()
            return

        check_dependencies()

        if not args.input:
            parser.print_help()
            return

        input_files: list[Path] = []
        seen_paths: set[Path] = set()
        skipped_dirs: list[Path] = []

        for pattern in args.input:
            expanded = glob.glob(pattern, recursive=True)
            if expanded:
                for match in expanded:
                    p = Path(match).resolve()
                    if p.is_dir():
                        count_before = len(input_files)
                        found = collect_media_from_dir(
                            p,
                            recursive=not args.no_recurse,
                            seen=seen_paths,
                        )
                        if found:
                            input_files.extend(found)
                            mode = "рекурсивно" if not args.no_recurse else "без рекурсии"
                            print(
                                f"📂 {p}: найдено {len(found)} медиафайл(ов) ({mode})"
                            )
                        else:
                            skipped_dirs.append(p)
                    elif p.is_file() and p not in seen_paths:
                        seen_paths.add(p)
                        input_files.append(p)
            else:
                path = Path(pattern).resolve()
                if path.exists():
                    if path.is_dir():
                        found = collect_media_from_dir(
                            path,
                            recursive=not args.no_recurse,
                            seen=seen_paths,
                        )
                        if found:
                            input_files.extend(found)
                            mode = "рекурсивно" if not args.no_recurse else "без рекурсии"
                            print(
                                f"📂 {path}: найдено {len(found)} медиафайл(ов) ({mode})"
                            )
                        else:
                            skipped_dirs.append(path)
                    elif path not in seen_paths:
                        seen_paths.add(path)
                        input_files.append(path)
                else:
                    print(f"❌ Файл/директория не найдены: {path}")

        for d in skipped_dirs:
            print(f"⚠️  Директория не содержит медиафайлов: {d}")

        if not input_files:
            raise ValidationError("Нет файлов для обработки")

        if args.output and len(input_files) > 1:
            raise ValidationError(
                "Опция -o/--output работает только с одним входным файлом"
            )

        if args.info:
            print_info(input_files)
            return

        track_filter = parse_track_filter(args.tracks)
        config = build_config(args, track_filter)
        check_model_name(config.model_name)

        swears = load_swears()
        swears_hash = compute_swears_hash(swears)
        matcher = build_swear_matcher(swears)

        print(f"\n🤖 Модель: {config.model_name}")
        gpu_slots = config.max_gpu_slots
        gpu_sem = ResourceSemaphore(
            "gpu_slot",
            slots=gpu_slots,
            label=f"GPU-слот (макс. {gpu_slots} параллельных)",
            enabled=gpu_slots > 0,
        )

        # Захватываем GPU-слот ПЕРЕД загрузкой модели и держим до конца.
        # Это гарантирует, что в GPU одновременно не более N моделей.
        with gpu_sem:
            model = load_whisper_model(config.model_name)

            file_results: list[FileResult] = []
            total_files = len(input_files)
            for i, input_path in enumerate(input_files, 1):
                if total_files > 1:
                    print(f"\n{'═'*60}")
                    print(f"📂 Файл [{i}/{total_files}]: {input_path.name}")
                    print("═" * 60)

                output_path = (
                    get_output_path(input_path, args.output)
                    if total_files == 1
                    else get_output_path(input_path, None)
                )
                media_type = probe_media_type(input_path)
                if media_type == "audio":
                    result = process_audio_file(
                        input_file=input_path,
                        output_file=output_path,
                        model=model,
                        matcher=matcher,
                        swears_hash=swears_hash,
                        config=config,
                    )
                else:
                    result = process_video_file(
                        input_file=input_path,
                        output_file=output_path,
                        model=model,
                        matcher=matcher,
                        swears_hash=swears_hash,
                        config=config,
                    )
                file_results.append(result)

        if total_files > 1:
            print(f"\n{'═'*60}")
            print("📊 ИТОГИ:")
            print("═" * 60)
            for r in file_results:
                icon = (
                    "✅"
                    if r.status == "ok"
                    else ("⚠️" if r.status == "partial_failed" else "❌")
                )
                print(f"  {icon} {Path(r.input_path).name}: {r.status}")

        report = build_run_report(file_results)
        print(
            f"\n📈 Всего: файлов={report['totals']['files_total']}, "
            f"ok={report['totals']['files_ok']}, "
            f"partial={report['totals']['files_partial_failed']}, "
            f"failed={report['totals']['files_failed']}"
        )
        print(
            f"   Матчей={report['totals']['matches_total']}, "
            f"зацензурено={report['totals']['censored_total']}, "
            f"время={report['totals']['duration_ms_total']/1000:.2f}s"
        )

        if config.report_json_path:
            write_report_json(config.report_json_path, report)

        has_fail = any(r.status == "failed" for r in file_results)
        has_partial = any(r.status == "partial_failed" for r in file_results)
        sys.exit(1 if (has_fail or has_partial) else 0)

    except KeyboardInterrupt:
        print("\n⛔ Остановлено пользователем (Ctrl+C). Очистка временных файлов...")
        cleanup_temp_files()
        sys.exit(130)
    except CensorErrorBase as exc:
        print(f"❌ Ошибка: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Непредвиденная ошибка: {type(exc).__name__}: {exc}")
        sys.exit(1)
    finally:
        cleanup_temp_files()


if __name__ == "__main__":
    main()
