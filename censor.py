#!/usr/bin/env python3
"""
Быстрая цензура мата в видео c GPU-ускорением (RTX 4060).

Использует faster-whisper (CTranslate2) для транскрипции —
примерно в 4-6 раз быстрее стандартного Whisper.

Зависимости:
    pip install faster-whisper numpy soundfile --break-system-packages
    # ffmpeg должен быть в PATH

Использование:
    python censor.py video.mkv
    python censor.py video.mkv --model large-v3
    python censor.py video.mkv --tracks 0,2
    python censor.py video.mkv --beep          # бип вместо тишины
    python censor.py video.mkv --info          # показать дорожки
"""

import argparse
import hashlib
import json
import subprocess
import sys
import shutil
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Ленивый импорт faster-whisper (для --info без загрузки модели)
WhisperModel = None


# ═══════════════════════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "medium"  # tiny, base, small, medium, large-v2, large-v3
SILENCE_THRESHOLD_DB = -50
BEEP_FREQ = 1000  # Гц
PADDING_MS = 50  # расширить цензуру на X мс с каждой стороны

# Расширения файлов
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.mov', '.webm', '.ts', '.m2ts', '.wmv'}

# Пути
SCRIPT_DIR = Path(__file__).parent
SWEARS_FILE = SCRIPT_DIR / "swears.txt"
SWEARS_FILE_ALT = Path.home() / ".config" / "censor" / "swears.txt"
CACHE_DIR = SCRIPT_DIR / "cache"


# ═══════════════════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioTrack:
    stream_index: int
    audio_index: int
    codec: str
    channels: int
    sample_rate: int
    title: str


@dataclass
class SwearMatch:
    start: float  # секунды
    end: float
    word: str


# ═══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str, prefix: str = ""):
    print(f"{prefix}{msg}")


def print_progress(current: float, total: float, prefix: str = "", width: int = 30):
    """Простой прогресс-бар в консоли."""
    if total <= 0:
        return
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r{prefix} [{bar}] {pct*100:.0f}%", end='', flush=True)
    if current >= total:
        print()


def is_audio_file(path: Path) -> bool:
    """Проверяет, является ли файл аудиофайлом."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Проверяет, является ли файл видеофайлом."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_audio_duration(path: Path) -> float:
    """Получает длительность аудио в секундах."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        return 0
    try:
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))
    except (ValueError, KeyError):
        return 0


def run_cmd(cmd: list[str], capture: bool = False, quiet: bool = True) -> subprocess.CompletedProcess:
    """Запуск команды с обработкой ошибок."""
    try:
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True)
        elif quiet:
            return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            return subprocess.run(cmd)
    except FileNotFoundError:
        print(f"❌ Не найдена команда: {cmd[0]}")
        sys.exit(1)


def run_ffmpeg_with_progress(cmd: list[str], duration: float, prefix: str = "") -> bool:
    """Запускает ffmpeg с отображением прогресса."""
    if duration <= 0:
        # Без прогресса
        return run_cmd(cmd, quiet=True).returncode == 0

    # Добавляем вывод прогресса
    cmd_with_progress = cmd.copy()
    # Вставляем после ffmpeg и перед входным файлом
    insert_pos = 1
    cmd_with_progress.insert(insert_pos, "-progress")
    cmd_with_progress.insert(insert_pos + 1, "pipe:1")

    try:
        process = subprocess.Popen(
            cmd_with_progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        current_time = 0.0
        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    time_ms = int(line.split("=")[1])
                    current_time = time_ms / 1_000_000  # микросекунды в секунды
                    print_progress(current_time, duration, prefix)
                except ValueError:
                    pass
            elif line == "progress=end":
                print_progress(duration, duration, prefix)

        process.wait()
        return process.returncode == 0

    except FileNotFoundError:
        print(f"❌ Не найдена команда: ffmpeg")
        return False


def get_file_hash(path: Path) -> str:
    """Быстрый хеш по метаданным файла."""
    stat = path.stat()
    data = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(data.encode()).hexdigest()[:10]


def get_cache_dir(input_file: Path) -> Path:
    """Папка кеша для файла."""
    h = get_file_hash(input_file)
    cache = CACHE_DIR / f"{input_file.stem}_{h}"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def load_swears() -> set[str]:
    """Загружает список мата."""
    swears_file = SWEARS_FILE if SWEARS_FILE.exists() else SWEARS_FILE_ALT

    if not swears_file.exists():
        print(f"❌ Файл swears.txt не найден!")
        print(f"   Ожидаемые пути:")
        print(f"   - {SWEARS_FILE}")
        print(f"   - {SWEARS_FILE_ALT}")
        sys.exit(1)

    swears = set()
    for line in swears_file.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            swears.add(word)

    log(f"📝 Загружено {len(swears)} слов из {swears_file.name}")
    return swears


# ═══════════════════════════════════════════════════════════════════════════════
# РАБОТА С ВИДЕО/АУДИО
# ═══════════════════════════════════════════════════════════════════════════════

def get_audio_tracks(input_file: Path) -> list[AudioTrack]:
    """Получает список аудиодорожек."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(input_file)]
    result = run_cmd(cmd, capture=True)

    if result.returncode != 0:
        print("❌ Ошибка ffprobe")
        sys.exit(1)

    info = json.loads(result.stdout)
    tracks = []
    audio_idx = 0

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            tracks.append(AudioTrack(
                stream_index=stream.get("index", 0),
                audio_index=audio_idx,
                codec=stream.get("codec_name", "aac"),
                channels=stream.get("channels", 2),
                sample_rate=int(stream.get("sample_rate", 48000)),
                title=stream.get("tags", {}).get("title", f"Track {audio_idx}")
            ))
            audio_idx += 1

    return tracks


def get_audio_info(input_file: Path) -> Optional[AudioTrack]:
    """Получает информацию об аудиофайле (для чистых аудио без видео)."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(input_file)]
    result = run_cmd(cmd, capture=True)

    if result.returncode != 0:
        return None

    try:
        info = json.loads(result.stdout)
    except (ValueError, KeyError):
        return None

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return AudioTrack(
                stream_index=stream.get("index", 0),
                audio_index=0,
                codec=stream.get("codec_name", "mp3"),
                channels=stream.get("channels", 2),
                sample_rate=int(stream.get("sample_rate", 44100)),
                title=input_file.stem
            )

    return None


def convert_audio_for_whisper(input_file: Path, output_wav: Path,
                               show_progress: bool = False, prefix: str = "") -> bool:
    """Конвертирует любой аудиофайл в WAV для Whisper (16kHz mono)."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file),
        "-ac", "1",           # моно
        "-ar", "16000",       # 16kHz
        "-acodec", "pcm_s16le",
        str(output_wav)
    ]
    if show_progress:
        duration = get_audio_duration(input_file)
        return run_ffmpeg_with_progress(cmd, duration, prefix)
    return run_cmd(cmd, quiet=True).returncode == 0


def convert_audio_full(input_file: Path, output_wav: Path,
                       show_progress: bool = False, prefix: str = "") -> bool:
    """Конвертирует аудиофайл в WAV с полным качеством."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file),
        "-acodec", "pcm_f32le",
        str(output_wav)
    ]
    if show_progress:
        duration = get_audio_duration(input_file)
        return run_ffmpeg_with_progress(cmd, duration, prefix)
    return run_cmd(cmd, quiet=True).returncode == 0


def extract_audio(input_file: Path, track: AudioTrack, output_wav: Path,
                  show_progress: bool = False, prefix: str = "") -> bool:
    """Извлекает аудиодорожку в WAV (оптимизировано для Whisper)."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file),
        "-map", f"0:a:{track.audio_index}",
        "-ac", "1",           # моно (Whisper работает с моно)
        "-ar", "16000",       # 16kHz (оптимально для Whisper)
        "-acodec", "pcm_s16le",
        str(output_wav)
    ]
    if show_progress:
        duration = get_audio_duration(input_file)
        return run_ffmpeg_with_progress(cmd, duration, prefix)
    return run_cmd(cmd, quiet=True).returncode == 0


def extract_audio_full(input_file: Path, track: AudioTrack, output_wav: Path,
                       show_progress: bool = False, prefix: str = "") -> bool:
    """Извлекает полное качество для обработки."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file),
        "-map", f"0:a:{track.audio_index}",
        "-acodec", "pcm_f32le",  # float32 для точной обработки
        str(output_wav)
    ]
    if show_progress:
        duration = get_audio_duration(input_file)
        return run_ffmpeg_with_progress(cmd, duration, prefix)
    return run_cmd(cmd, quiet=True).returncode == 0


def is_silent(wav_path: Path) -> bool:
    """Проверяет, тихая ли дорожка."""
    cmd = ["ffmpeg", "-i", str(wav_path), "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stderr.split('\n'):
        if 'max_volume' in line:
            try:
                max_vol = float(line.split('max_volume:')[1].split('dB')[0].strip())
                return max_vol < SILENCE_THRESHOLD_DB
            except:
                pass
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# ТРАНСКРИПЦИЯ (FASTER-WHISPER)
# ═══════════════════════════════════════════════════════════════════════════════

def load_whisper_model(model_name: str):
    """Загружает модель Whisper с CUDA (без повторного скачивания)."""
    global WhisperModel
    if WhisperModel is None:
        from faster_whisper import WhisperModel as WM
        WhisperModel = WM

    models_dir = CACHE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем, есть ли модель локально
    local_model = models_dir / f"models--Systran--faster-whisper-{model_name}"
    if local_model.exists():
        log(f"🤖 Загрузка модели {model_name} (локальная, CUDA float16)...")
    else:
        log(f"🤖 Скачивание и загрузка модели {model_name} (CUDA float16)...")

    return WhisperModel(
        model_name,
        device="cuda",
        compute_type="float16",
        download_root=str(models_dir),
        local_files_only=local_model.exists()  # не лезть в сеть, если модель есть
    )

def transcribe(model, audio_path: Path, language: str = "ru", show_progress: bool = True) -> list[dict]:
    """Транскрибирует аудио с word-level timestamps и прогрессом."""
    # Получаем длительность для прогресса
    duration = get_audio_duration(audio_path) if show_progress else 0

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,  # фильтр тишины — ускоряет
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )

    words = []
    for segment in segments:
        # Обновляем прогресс
        if show_progress and duration > 0:
            print_progress(segment.end, duration, "      🎤 Транскрипция")

        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })

    return words


# ═══════════════════════════════════════════════════════════════════════════════
# ПОИСК МАТА
# ═══════════════════════════════════════════════════════════════════════════════

def find_swears(words: list[dict], swears: set[str]) -> list[SwearMatch]:
    """Находит матерные слова в транскрипции."""
    matches = []

    # Строим regex для каждого слова из словаря
    # Учитываем возможные формы слова (окончания)
    patterns = []
    for swear in swears:
        # Базовое слово + любые окончания
        pattern = re.escape(swear)
        patterns.append(pattern)

    combined_pattern = re.compile(
        r'\b(' + '|'.join(patterns) + r')[а-яёa-z]*\b',
        re.IGNORECASE
    )

    for w in words:
        clean_word = re.sub(r'[^\w]', '', w["word"].lower())
        if combined_pattern.search(clean_word):
            matches.append(SwearMatch(
                start=max(0, w["start"] - PADDING_MS / 1000),
                end=w["end"] + PADDING_MS / 1000,
                word=w["word"]
            ))

    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# ОБРАБОТКА АУДИО
# ═══════════════════════════════════════════════════════════════════════════════

def generate_beep(duration_sec: float, sample_rate: int, channels: int) -> np.ndarray:
    """Генерирует бип заданной длительности."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    beep = 0.5 * np.sin(2 * np.pi * BEEP_FREQ * t)

    # Fade in/out для плавности
    fade_samples = min(int(0.01 * sample_rate), len(beep) // 4)
    if fade_samples > 0:
        beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
        beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    if channels > 1:
        beep = np.tile(beep.reshape(-1, 1), (1, channels))

    return beep


def censor_audio(audio_path: Path, output_path: Path, matches: list[SwearMatch],
                 sample_rate: int, channels: int, use_beep: bool = False,
                 show_progress: bool = True) -> int:
    """Накладывает цензуру на аудио."""
    import soundfile as sf

    # Читаем аудио
    audio, sr = sf.read(str(audio_path), dtype='float32')

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
        actual_channels = 1
    else:
        actual_channels = audio.shape[1]

    censored_count = 0
    total_matches = len(matches)

    for i, match in enumerate(matches):
        if show_progress and total_matches > 10:
            print_progress(i + 1, total_matches, "      🔇 Цензура")

        start_sample = int(match.start * sr)
        end_sample = int(match.end * sr)

        # Границы
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if start_sample >= end_sample:
            continue

        duration = (end_sample - start_sample) / sr

        if use_beep:
            beep = generate_beep(duration, sr, actual_channels)
            if len(beep) != end_sample - start_sample:
                beep = np.resize(beep, (end_sample - start_sample, actual_channels))
            audio[start_sample:end_sample] = beep
        else:
            # Тишина
            audio[start_sample:end_sample] = 0

        censored_count += 1

    # Записываем
    if actual_channels == 1:
        audio = audio.flatten()

    sf.write(str(output_path), audio, sr, subtype='FLOAT')

    return censored_count


def encode_audio(input_wav: Path, output_file: Path, track: AudioTrack,
                 show_progress: bool = False, prefix: str = "") -> bool:
    """Кодирует WAV в оригинальный формат."""
    codec_map = {
        "aac": ["-c:a", "aac", "-b:a", "192k"],
        "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
        "opus": ["-c:a", "libopus", "-b:a", "128k"],
        "vorbis": ["-c:a", "libvorbis", "-b:a", "192k"],
        "flac": ["-c:a", "flac"],
        "ac3": ["-c:a", "ac3", "-b:a", "384k"],
        "eac3": ["-c:a", "eac3", "-b:a", "384k"],
        "dts": ["-c:a", "dca", "-b:a", "768k"],
    }

    codec_args = codec_map.get(track.codec, ["-c:a", "aac", "-b:a", "192k"])

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_wav),
        "-ar", str(track.sample_rate),
        "-ac", str(track.channels),
        *codec_args,
        str(output_file)
    ]

    if show_progress:
        duration = get_audio_duration(input_wav)
        return run_ffmpeg_with_progress(cmd, duration, prefix)
    return run_cmd(cmd, quiet=True).returncode == 0


def copy_audio_track(input_file: Path, track: AudioTrack, output_file: Path) -> bool:
    """Копирует дорожку без изменений."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file),
        "-map", f"0:a:{track.audio_index}",
        "-c:a", "copy",
        str(output_file)
    ]
    return run_cmd(cmd, quiet=True).returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# СБОРКА ВИДЕО
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_video(input_file: Path, audio_files: list[Path],
                   output_file: Path, tracks: list[AudioTrack]) -> bool:
    """Собирает финальное видео."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats",
           "-i", str(input_file)]

    for audio in audio_files:
        cmd.extend(["-i", str(audio)])

    # Маппинг: видео и субтитры из оригинала, аудио из обработанных
    cmd.extend(["-map", "0:v?"])

    for i in range(len(audio_files)):
        cmd.extend(["-map", f"{i+1}:a:0"])

    cmd.extend(["-map", "0:s?"])  # субтитры
    cmd.extend(["-c:v", "copy", "-c:s", "copy", "-c:a", "copy"])

    # Метаданные дорожек
    for i, track in enumerate(tracks):
        if track.title:
            cmd.extend([f"-metadata:s:a:{i}", f"title={track.title}"])

    cmd.append(str(output_file))

    return run_cmd(cmd, quiet=False).returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ЛОГИКА
# ═══════════════════════════════════════════════════════════════════════════════

def process_track(model, input_file: Path, track: AudioTrack, cache_dir: Path,
                  swears: set[str], use_beep: bool) -> tuple[Path, int]:
    """Обрабатывает одну аудиодорожку."""

    idx = track.audio_index

    # Пути
    wav_whisper = cache_dir / f"track_{idx}_16k.wav"
    wav_full = cache_dir / f"track_{idx}_full.wav"
    wav_censored = cache_dir / f"track_{idx}_censored.wav"
    final_audio = cache_dir / f"track_{idx}_final.mka"
    transcript_cache = cache_dir / f"track_{idx}_transcript.json"
    skip_marker = cache_dir / f"track_{idx}_skip"

    # Уже готово?
    if final_audio.exists():
        log(f"💾 Кеш", prefix="      ")
        return final_audio, -1  # -1 = из кеша

    # Пропущена ранее?
    if skip_marker.exists():
        log(f"🔇 Тихая (кеш)", prefix="      ")
        if not final_audio.exists():
            copy_audio_track(input_file, track, final_audio)
        return final_audio, 0

    # 1. Извлекаем для Whisper (16kHz mono)
    if not wav_whisper.exists():
        log(f"📤 Извлечение...", prefix="      ")
        if not extract_audio(input_file, track, wav_whisper):
            raise RuntimeError("Ошибка извлечения")

    # 2. Проверка на тишину
    if is_silent(wav_whisper):
        log(f"🔇 Тихая дорожка", prefix="      ")
        skip_marker.touch()
        copy_audio_track(input_file, track, final_audio)
        return final_audio, 0

    # 3. Транскрипция
    words = []
    if transcript_cache.exists():
        log(f"📝 Транскрипт из кеша", prefix="      ")
        words = json.loads(transcript_cache.read_text(encoding="utf-8"))
    else:
        log(f"🎤 Транскрипция...", prefix="      ")
        words = transcribe(model, wav_whisper)
        transcript_cache.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")

    # 4. Поиск мата
    matches = find_swears(words, swears)
    log(f"🔍 Найдено: {len(matches)} слов", prefix="      ")

    if not matches:
        # Нет мата — копируем как есть
        copy_audio_track(input_file, track, final_audio)
        return final_audio, 0

    # 5. Извлекаем полное качество
    if not wav_full.exists():
        log(f"📤 Извлечение (полное качество)...", prefix="      ")
        if not extract_audio_full(input_file, track, wav_full):
            raise RuntimeError("Ошибка извлечения")

    # 6. Цензурим
    log(f"🔇 Цензура...", prefix="      ")
    censored = censor_audio(wav_full, wav_censored, matches,
                            track.sample_rate, track.channels, use_beep)

    # 7. Кодируем обратно
    log(f"🔄 Кодирование ({track.codec})...", prefix="      ")
    if not encode_audio(wav_censored, final_audio, track):
        raise RuntimeError("Ошибка кодирования")

    return final_audio, censored


def process_audio_file(input_file: Path, output_file: Path, model,
                       swears: set[str], use_beep: bool) -> bool:
    """Обрабатывает чистый аудиофайл (без видео)."""

    print(f"\n🎵 Вход:  {input_file}")
    print(f"📁 Выход: {output_file}")
    print()

    # Кеш
    cache_dir = get_cache_dir(input_file)
    log(f"💾 Кеш: {cache_dir}")

    # Получаем информацию об аудио
    audio_info = get_audio_info(input_file)
    if not audio_info:
        print("❌ Не удалось прочитать аудиофайл!")
        return False

    print(f"📊 {audio_info.codec}, {audio_info.channels}ch, {audio_info.sample_rate}Hz")

    # Пути
    wav_whisper = cache_dir / "audio_16k.wav"
    wav_full = cache_dir / "audio_full.wav"
    wav_censored = cache_dir / "audio_censored.wav"
    transcript_cache = cache_dir / "transcript.json"

    # 1. Конвертируем для Whisper (16kHz mono)
    if not wav_whisper.exists():
        log("📤 Конвертация для транскрипции...")
        if not convert_audio_for_whisper(input_file, wav_whisper,
                                         show_progress=True, prefix="   📤 Конвертация"):
            print("❌ Ошибка конвертации")
            return False

    # 2. Проверка на тишину
    if is_silent(wav_whisper):
        log("🔇 Тихий файл — копируем как есть")
        shutil.copy(input_file, output_file)
        return True

    # 3. Транскрипция
    words = []
    if transcript_cache.exists():
        log("📝 Транскрипт из кеша")
        words = json.loads(transcript_cache.read_text(encoding="utf-8"))
    else:
        log("🎤 Транскрипция...")
        words = transcribe(model, wav_whisper)
        transcript_cache.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")

    # 4. Поиск мата
    matches = find_swears(words, swears)
    log(f"🔍 Найдено: {len(matches)} слов")

    if not matches:
        log("✅ Мата нет — копируем как есть")
        shutil.copy(input_file, output_file)
        return True

    # 5. Конвертируем в полное качество
    if not wav_full.exists():
        log("📤 Конвертация (полное качество)...")
        if not convert_audio_full(input_file, wav_full,
                                  show_progress=True, prefix="   📤 Конвертация"):
            print("❌ Ошибка конвертации")
            return False

    # 6. Цензурим
    log("🔇 Цензура...")
    censored = censor_audio(wav_full, wav_censored, matches,
                            audio_info.sample_rate, audio_info.channels, use_beep)

    # 7. Кодируем в исходный формат
    log(f"🔄 Кодирование ({audio_info.codec})...")
    if not encode_audio(wav_censored, output_file, audio_info,
                        show_progress=True, prefix="   🔄 Кодирование"):
        print("❌ Ошибка кодирования")
        return False

    print(f"\n{'═'*50}")
    print("✅ ГОТОВО!")
    print(f"📁 {output_file}")
    print(f"🔇 Зацензурено: {censored} слов")
    print('═'*50)

    return True


def process_video_file(input_file: Path, output_file: Path, model,
                       swears: set[str], track_filter: Optional[list[int]],
                       use_beep: bool) -> bool:
    """Обрабатывает видеофайл."""

    print(f"\n🎬 Вход:  {input_file}")
    print(f"📁 Выход: {output_file}")
    print()

    # Кеш
    cache_dir = get_cache_dir(input_file)
    log(f"💾 Кеш: {cache_dir}")

    # Получаем дорожки
    tracks = get_audio_tracks(input_file)
    if not tracks:
        print("❌ Аудиодорожки не найдены!")
        return False

    print(f"\n📊 Аудиодорожек: {len(tracks)}")
    for t in tracks:
        print(f"   [{t.audio_index}] {t.title} ({t.codec}, {t.channels}ch, {t.sample_rate}Hz)")

    # Фильтр
    if track_filter:
        tracks = [t for t in tracks if t.audio_index in track_filter]
        print(f"\n⚙️  Выбраны дорожки: {track_filter}")

    # Обрабатываем дорожки
    processed_files = []
    total_censored = 0

    for i, track in enumerate(tracks, 1):
        print(f"\n{'─'*50}")
        print(f"🎵 [{i}/{len(tracks)}] Дорожка {track.audio_index}: {track.title}")

        try:
            audio_file, censored = process_track(
                model, input_file, track, cache_dir, swears, use_beep
            )
            processed_files.append(audio_file)
            if censored > 0:
                total_censored += censored
                log(f"✅ Зацензурено: {censored}", prefix="      ")
            elif censored == 0:
                log("✅ Готово (мата нет)", prefix="      ")
            else:
                log("✅ Готово (из кеша)", prefix="      ")
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")
            return False

    # Сборка
    print(f"\n{'─'*50}")
    print("📦 Сборка видео...")

    if not assemble_video(input_file, processed_files, output_file, tracks):
        print("❌ Ошибка сборки")
        return False

    print(f"\n{'═'*50}")
    print("✅ ГОТОВО!")
    print(f"📁 {output_file}")
    print(f"🔇 Всего зацензурено: {total_censored} слов")
    print('═'*50)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Быстрая цензура мата в видео и аудио (GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s video.mkv                    # видео, все дорожки, модель medium
  %(prog)s podcast.mp3                  # аудиофайл
  %(prog)s *.mkv *.mp3                  # несколько файлов
  %(prog)s video.mkv -m large-v3        # точнее, но медленнее
  %(prog)s video.mkv -t 0,2             # только дорожки 0 и 2
  %(prog)s video.mkv --beep             # бип вместо тишины
  %(prog)s video.mkv --info             # показать дорожки
  %(prog)s --clear-cache                # очистить весь кеш
        """
    )

    parser.add_argument("input", nargs="*", help="Входные файлы (видео или аудио)")
    parser.add_argument("-o", "--output", help="Выходной файл (только для одного входного)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Модель Whisper (default: {DEFAULT_MODEL})")
    parser.add_argument("-t", "--tracks", help="Дорожки для видео (например: 0,2,4)")
    parser.add_argument("--beep", action="store_true", help="Бип вместо тишины")
    parser.add_argument("--info", action="store_true", help="Показать дорожки")
    parser.add_argument("--clear-cache", action="store_true", help="Очистить кеш (без моделей)")
    parser.add_argument("--clear-models", action="store_true", help="Очистить скачанные модели")

    args = parser.parse_args()

    # Очистка кеша (без моделей)
    if args.clear_cache:
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
        return

    # Очистка моделей
    if args.clear_models:
        models_dir = CACHE_DIR / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
            print(f"🗑️  Модели удалены: {models_dir}")
        else:
            print("Модели не найдены")
        return

    if not args.input:
        parser.print_help()
        return

    # Собираем и проверяем входные файлы
    input_files = []
    for pattern in args.input:
        path = Path(pattern).resolve()
        if path.exists():
            input_files.append(path)
        else:
            print(f"❌ Файл не найден: {path}")

    if not input_files:
        print("❌ Нет файлов для обработки!")
        sys.exit(1)

    # Инфо (только для первого файла)
    if args.info:
        for input_path in input_files:
            print(f"\n📁 {input_path}")
            if is_audio_file(input_path):
                info = get_audio_info(input_path)
                if info:
                    print(f"    🎵 Аудио: {info.codec}, {info.channels}ch, {info.sample_rate}Hz")
            else:
                tracks = get_audio_tracks(input_path)
                for t in tracks:
                    print(f"    [{t.audio_index}] {t.title}")
                    print(f"        {t.codec}, {t.channels}ch, {t.sample_rate}Hz")
        return

    # Проверка на -o при нескольких файлах
    if args.output and len(input_files) > 1:
        print("❌ Опция -o/--output работает только с одним входным файлом!")
        sys.exit(1)

    # Фильтр дорожек
    track_filter = None
    if args.tracks:
        track_filter = [int(x.strip()) for x in args.tracks.split(",")]

    # Загружаем словарь и модель один раз
    swears = load_swears()
    print(f"\n🤖 Модель: {args.model}")
    model = load_whisper_model(args.model)

    # Обрабатываем файлы
    results = []
    total_files = len(input_files)

    for i, input_path in enumerate(input_files, 1):
        if total_files > 1:
            print(f"\n{'═'*60}")
            print(f"📂 Файл [{i}/{total_files}]: {input_path.name}")
            print('═'*60)

        # Определяем выходной путь
        if args.output and total_files == 1:
            output_path = Path(args.output).resolve()
        else:
            output_path = input_path.parent / f"{input_path.stem}_censored{input_path.suffix}"

        # Обрабатываем в зависимости от типа
        if is_audio_file(input_path):
            success = process_audio_file(input_path, output_path, model, swears, args.beep)
        else:
            success = process_video_file(input_path, output_path, model, swears,
                                         track_filter, args.beep)

        results.append((input_path, success))

    # Итоги при нескольких файлах
    if total_files > 1:
        print(f"\n{'═'*60}")
        print("📊 ИТОГИ:")
        print('═'*60)
        success_count = sum(1 for _, s in results if s)
        for path, success in results:
            status = "✅" if success else "❌"
            print(f"  {status} {path.name}")
        print(f"\n  Успешно: {success_count}/{total_files}")

    # Код выхода
    all_success = all(s for _, s in results)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
