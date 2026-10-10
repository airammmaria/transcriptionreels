#!/usr/bin/env python3
"""Instagram Reel -> локальная транскрипция (yt-dlp + FFmpeg + faster-whisper).

Использование:
    python3 transcribe_reel.py <ссылка> <номер>
    python3 transcribe_reel.py "https://www.instagram.com/reel/XXXX/" 01

Результат: Reel_<номер>_Transcript.txt в текущей папке.
Перевод не выполняется, язык определяется автоматически.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from faster_whisper import WhisperModel

MODEL = "large-v3"
# Сегменты с низкой уверенностью помечаются как [неразборчиво], а не угадываются.
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6
# Пауза между фразами (сек), после которой начинается новый абзац.
PARAGRAPH_GAP = 1.5


def download(url: str, workdir: Path) -> Path:
    subprocess.run(
        ["yt-dlp", "--no-playlist", "-f", "best", "-o", str(workdir / "video.%(ext)s"), url],
        check=True,
    )
    return next(workdir.glob("video.*"))


def extract_audio(video: Path, workdir: Path) -> Path:
    audio = workdir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", str(audio)],
        check=True,
    )
    return audio


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def transcribe(audio: Path):
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio),
        task="transcribe",  # без перевода
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,  # снижает риск «галлюцинаций» повторами
    )
    paragraphs, current, last_end = [], [], None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.avg_logprob < MIN_AVG_LOGPROB or seg.no_speech_prob > MAX_NO_SPEECH_PROB:
            text = "[неразборчиво]"
        if last_end is not None and seg.start - last_end > PARAGRAPH_GAP and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        last_end = seg.end
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs, info


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    url, num = sys.argv[1], sys.argv[2]
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        audio = extract_audio(download(url, workdir), workdir)
        paragraphs, info = transcribe(audio)
    out = Path(f"Reel_{num}_Transcript.txt")
    out.write_text(
        f"Ссылка на оригинал: {url}\n"
        f"Язык: {info.language} (уверенность {info.language_probability:.2f})\n"
        f"Длительность: {fmt_duration(info.duration)}\n\n"
        "ПОЛНАЯ ТРАНСКРИПЦИЯ:\n\n" + "\n\n".join(paragraphs) + "\n",
        encoding="utf-8",
    )
    print(f"Готово: {out} за {time.monotonic() - started:.0f} c")


if __name__ == "__main__":
    main()
