from __future__ import annotations

import base64
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string
from PIL import Image
from werkzeug.utils import secure_filename

from infer import (
    get_device,
    load_checkpoint_and_cfg,
    prepare_input_tensors,
    prepare_frames_from_dir,
    extract_face_frames_from_video,
    run_inference,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MODEL_SEARCH_DIRS = [
    Path("./experiments"),
    Path("./experiments_dfdc02"),
    Path("./experiments_dfd01"),
]

INLINE_VIDEO_MAX_MB = 25
MAX_PREVIEW_FRAMES = 16
LANGS = {"ru", "en"}


TRANSLATIONS = {
    "ru": {
        "page_title": "Deepfake Detection MVP",
        "hero_subtitle": "Загрузите одно видео или одну папку с кадрами, выберите найденный checkpoint и выполните video-level deepfake inference с сохранением JSON-результата.",
        "hero_badge": "Flask интерфейс · single-sample inference",

        "run_inference": "Запуск инференса",
        "run_subtitle": "Минимальный и понятный сценарий ввода, стабильное поведение backend и чистый результат для smoke-test и демо ВКР.",

        "model": "Модель",
        "model_hint": "Короткие human-readable названия. Рекомендуемый checkpoint выбирается автоматически.",
        "no_models_found": "Модели не найдены",

        "input_mode": "Режим входа",
        "input_mode_hint": "Используйте raw video или одну папку с заранее извлечёнными кадрами.",

        "device": "Устройство",
        "device_hint": "Auto выбирает лучшее доступное устройство. Неподдерживаемые операции на MPS автоматически переводятся на CPU.",

        "clip_length": "Длина клипа для инференса",
        "clip_length_hint": "Это значение берётся из config выбранного checkpoint. Свободное переключение 16 / 20 / 40 / 60 намеренно отключено ради корректности.",
        "clip_length_unknown": "Будет прочитано из checkpoint после выбора",

        "upload_video": "Загрузить видео",
        "upload_video_hint": "Поддерживаемые форматы: mp4, avi, mov, mkv, webm, m4v. Одиночная картинка не подходит для video-level inference.",
        "no_file_selected": "Файл не выбран.",

        "upload_frames": "Загрузить папку с кадрами",
        "upload_frames_hint": "Одна последовательность на одну папку. Поддерживаются JPG, PNG и BMP.",
        "no_frames_selected": "Кадры не выбраны.",

        "run_button": "Запустить инференс",
        "running_button": "Идёт инференс...",

        "session_summary": "Сводка сессии",
        "session_subtitle": "Текущий выбор и важные ограничения запуска.",
        "selected_model": "Выбранная модель",
        "selected_model_sub": "Рекомендуемый DFDC02 full adaptive checkpoint автоматически приоритизируется, если доступен.",
        "selected_input_mode": "Режим входа",
        "selected_input_mode_sub": "Raw video использует internal MTCNN face extraction. Папка кадров предполагает, что preprocessing уже выполнен.",
        "requested_device": "Запрошенное устройство",
        "requested_device_sub": "Кроссплатформенный inference сохранён. Flask добавляет только безопасный fallback для нестабильных MPS runtime-case.",
        "current_source": "Текущий источник",
        "current_source_sub": "Интерфейс не требует ручного ввода путей и соответствует текущему пайплайну ВКР.",
        "important_limits": "Важные ограничения",
        "important_limits_value": "Face-centric · главное лицо · максимум 300 MB",
        "important_limits_sub": "Если в raw video не найдено лицо, инференс завершится понятной ошибкой. В multi-face сценах сейчас используется самое крупное лицо.",

        "info_note": "Важно: текущая длина клипа для инференса фиксируется config выбранного checkpoint. Свободное переключение 16 / 20 / 40 / 60 намеренно отключено, чтобы не ломать корректность thesis-пайплайна.",

        "completed_successfully": "Успешно завершено",
        "completed_with_fallback": "Завершено с fallback устройства",

        "inference_result": "Результат инференса",
        "result_subtitle": "Краткий итог, метаданные запуска и структурированный вывод для демо.",

        "prediction": "Предсказание",
        "probability_fake": "Вероятность fake",
        "device_used": "Использованное устройство",
        "frames_used": "Использовано кадров",

        "fake_detected": "Fake обнаружен",
        "likely_real": "Скорее real",

        "run_details": "Детали запуска",
        "model_type": "Тип модели",
        "fusion_type": "Тип fusion",
        "dataset": "Датасет",
        "threshold": "Порог",
        "final_device": "Финальное устройство",
        "fallback": "Fallback",

        "input_details": "Детали входа",
        "input_type": "Тип входа",
        "source_file": "Исходный файл",
        "source_path": "Путь к источнику",
        "source_frames": "Исходных кадров",
        "preprocessing": "Preprocessing",

        "fusion_weights": "Fusion weights",
        "tested_video": "Тестируемое видео",
        "video_preview_hidden": "Inline preview видео скрыт, потому что файл слишком большой для безопасного встраивания.",
        "video_preview_note": "Видео показывается inline только если размер файла достаточно мал для безопасного preview.",

        "frames_used_for_inference": "Кадры, реально использованные для инференса",
        "preview_selected_frame": "Просмотр выбранного sampled frame",

        "raw_result_details": "Сырые детали результата",
        "fallback_reason": "Причина fallback",

        "saved_json_path": "Путь к сохранённому JSON",
        "saved_json_subtitle": "Result payload сохранён рядом с выбранным checkpoint.",

        "lang_ru": "Рус",
        "lang_en": "Eng",

        "video_label": "video",
        "frames_dir_label": "frames_dir",

        "unsupported_video_format": "Неподдерживаемый формат видео",
        "supported_video_formats": "Поддерживаемые форматы",
        "supported_image_formats": "Поддерживаемые форматы изображений",
        "no_video_uploaded": "Видео не загружено.",
        "no_frames_uploaded": "Папка с кадрами не загружена.",
        "no_supported_images": "В выбранной папке не найдено ни одного поддерживаемого изображения.",
        "no_frames_after_upload": "После загрузки не найдено ни одного кадра.",
        "unknown_input_mode": "Неизвестный input_mode",
        "checkpoint_not_found": "Checkpoint не найден",
        "no_available_model": "Не найдено ни одной доступной модели.",
        "upload_too_large": "Файл слишком большой. Максимально допустимый размер — 300 MB.",
        "single_image_not_supported": "Одиночное изображение здесь не поддерживается. Используйте raw video или папку кадров для video-level inference.",
        "no_face_detected": "В загруженном видео не найдено лицо. Загрузите ролик, где хотя бы одно лицо хорошо видно.",
        "invalid_video_decode": "Загруженный файл не удалось корректно декодировать как валидное видео. Проверьте файл и попробуйте другой.",
        "selected": "Выбрано",
        "selected_files": "Выбрано файлов",
        "frames_selected": "Кадров выбрано",
        "not_selected": "Не выбрано",
        "sampled_frame": "Sampled frame",
        "used_face_crop": "Used face crop",
    },
    "en": {
        "page_title": "Deepfake Detection MVP",
        "hero_subtitle": "Upload one video or one frames folder, choose a discovered checkpoint, and run video-level deepfake inference with saved JSON output.",
        "hero_badge": "Flask interface · single-sample inference",

        "run_inference": "Run inference",
        "run_subtitle": "Minimal input flow, stable backend behavior, and clean output for smoke-testing and thesis demo.",

        "model": "Model",
        "model_hint": "Short human-readable labels. Recommended checkpoint is selected automatically.",
        "no_models_found": "No models found",

        "input_mode": "Input mode",
        "input_mode_hint": "Use raw video or one folder of pre-extracted frames.",

        "device": "Device",
        "device_hint": "Auto selects the best available device. Unsupported MPS operations automatically fall back to CPU.",

        "clip_length": "Inference clip length",
        "clip_length_hint": "This value is read from the selected checkpoint config. Free 16 / 20 / 40 / 60 switching is intentionally disabled for correctness.",
        "clip_length_unknown": "Will be read from checkpoint after selection",

        "upload_video": "Upload video",
        "upload_video_hint": "Supported formats: mp4, avi, mov, mkv, webm, m4v. A single image is not valid for video-level inference.",
        "no_file_selected": "No file selected.",

        "upload_frames": "Upload frames folder",
        "upload_frames_hint": "One clip per folder. JPG, PNG and BMP files are supported.",
        "no_frames_selected": "No frames selected.",

        "run_button": "Run inference",
        "running_button": "Running inference...",

        "session_summary": "Session summary",
        "session_subtitle": "Current selection and important run constraints.",
        "selected_model": "Selected model",
        "selected_model_sub": "Recommended DFDC02 full adaptive checkpoint is prioritized automatically when available.",
        "selected_input_mode": "Input mode",
        "selected_input_mode_sub": "Raw video uses internal MTCNN face extraction. Frames folder assumes preprocessing is already done.",
        "requested_device": "Requested device",
        "requested_device_sub": "Cross-platform inference is preserved. Flask adds only safe fallback behavior for unstable MPS runtime cases.",
        "current_source": "Current source",
        "current_source_sub": "The interface avoids manual path entry and stays aligned with your thesis inference flow.",
        "important_limits": "Important limits",
        "important_limits_value": "Face-centric · dominant face · 300 MB max upload",
        "important_limits_sub": "If no face is detected in raw video, inference stops with a clear error. Multi-face scenes currently use the largest detected face.",

        "info_note": "Important: current inference clip length is fixed by the selected checkpoint config. Free 16 / 20 / 40 / 60 switching is intentionally disabled to keep the thesis pipeline correct.",

        "completed_successfully": "Completed successfully",
        "completed_with_fallback": "Completed with device fallback",

        "inference_result": "Inference result",
        "result_subtitle": "Prediction summary, run metadata, and structured output for demo presentation.",

        "prediction": "Prediction",
        "probability_fake": "Probability fake",
        "device_used": "Device used",
        "frames_used": "Frames used",

        "fake_detected": "Fake detected",
        "likely_real": "Likely real",

        "run_details": "Run details",
        "model_type": "Model type",
        "fusion_type": "Fusion type",
        "dataset": "Dataset",
        "threshold": "Threshold",
        "final_device": "Final device",
        "fallback": "Fallback",

        "input_details": "Input details",
        "input_type": "Input type",
        "source_file": "Source file",
        "source_path": "Source path",
        "source_frames": "Source frames",
        "preprocessing": "Preprocessing",

        "fusion_weights": "Fusion weights",
        "tested_video": "Tested video",
        "video_preview_hidden": "Inline video preview is hidden because the file is too large for safe embedding.",
        "video_preview_note": "The video is shown inline only when the file is small enough for safe preview.",

        "frames_used_for_inference": "Frames actually used for inference",
        "preview_selected_frame": "Preview selected sampled frame",

        "raw_result_details": "Raw result details",
        "fallback_reason": "Fallback reason",

        "saved_json_path": "Saved JSON path",
        "saved_json_subtitle": "The result payload was written next to the selected checkpoint.",

        "lang_ru": "Рус",
        "lang_en": "Eng",

        "video_label": "video",
        "frames_dir_label": "frames_dir",

        "unsupported_video_format": "Unsupported video format",
        "supported_video_formats": "Supported formats",
        "supported_image_formats": "Supported image formats",
        "no_video_uploaded": "No video was uploaded.",
        "no_frames_uploaded": "No frames folder was uploaded.",
        "no_supported_images": "No supported image files were found in the selected folder.",
        "no_frames_after_upload": "No image frames were found after upload.",
        "unknown_input_mode": "Unknown input_mode",
        "checkpoint_not_found": "Checkpoint not found",
        "no_available_model": "No available model was found.",
        "upload_too_large": "Upload is too large. Maximum allowed size is 300 MB.",
        "single_image_not_supported": "Single image input is not supported here. Use a raw video file or a folder of frames for video-level inference.",
        "no_face_detected": "No face was detected in the uploaded video. Please upload a clip with at least one clearly visible face.",
        "invalid_video_decode": "The uploaded file could not be decoded as a valid video. Please check the file and try another one.",
        "selected": "Selected",
        "selected_files": "Selected files",
        "frames_selected": "Frames selected",
        "not_selected": "Not selected",
        "sampled_frame": "Sampled frame",
        "used_face_crop": "Used face crop",
    },
}


HTML = """
<!doctype html>
<html lang="{{ lang }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ tr.page_title }}</title>
    <style>
        :root {
            --bg: #f4f7fb;
            --surface: #ffffff;
            --surface-2: #f8fafc;
            --text: #0f172a;
            --muted: #667085;
            --border: #d9e2ec;
            --border-strong: #c7d1dc;
            --shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            --radius: 18px;

            --navy: #0a1631;
            --navy-2: #14254c;

            --green-bg: #ecfdf3;
            --green-text: #027a48;
            --green-border: #abefc6;

            --red-bg: #fef3f2;
            --red-text: #b42318;
            --red-border: #fecdca;

            --amber-bg: #fffaeb;
            --amber-text: #b54708;
            --amber-border: #fedf89;

            --blue-bg: #eff8ff;
            --blue-text: #175cd3;
            --blue-border: #b2ddff;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.9), rgba(255,255,255,0) 35%),
                linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%);
        }

        .page {
            max-width: 1180px;
            margin: 0 auto;
            padding: 24px 18px 42px;
        }

        .hero {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: flex-start;
            margin-bottom: 18px;
        }

        .hero-right {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        .lang-switch {
            display: inline-flex;
            border: 1px solid var(--border);
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.86);
        }

        .lang-link {
            padding: 9px 12px;
            text-decoration: none;
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
            line-height: 1;
            border-right: 1px solid var(--border);
        }

        .lang-link:last-child {
            border-right: 0;
        }

        .lang-link.active {
            background: var(--navy);
            color: #fff;
        }

        .title {
            margin: 0 0 8px;
            font-size: 30px;
            line-height: 1.06;
            letter-spacing: -0.03em;
        }

        .subtitle {
            margin: 0;
            max-width: 820px;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.55;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.86);
            color: var(--muted);
            font-size: 13px;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(330px, 0.85fr);
            gap: 20px;
            align-items: start;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .card-inner {
            padding: 24px;
        }

        .section-title {
            margin: 0 0 6px;
            font-size: 19px;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            margin: 0 0 20px;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.5;
        }

        .field { margin-bottom: 16px; }

        label {
            display: block;
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 14px;
        }

        select,
        input[type=file],
        input[type=range],
        input[type=text] {
            width: 100%;
            min-height: 46px;
            padding: 11px 13px;
            border-radius: 12px;
            border: 1px solid var(--border-strong);
            background: #fff;
            color: var(--text);
            font-size: 15px;
        }

        select:focus,
        input[type=file]:focus,
        input[type=range]:focus,
        input[type=text]:focus {
            outline: none;
            border-color: #98b5ff;
            box-shadow: 0 0 0 4px rgba(23, 92, 211, 0.12);
        }

        .hint {
            margin-top: 7px;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .upload-box {
            padding: 14px;
            background: var(--surface-2);
            border: 1px dashed var(--border-strong);
            border-radius: 14px;
        }

        .upload-meta {
            margin-top: 8px;
            color: var(--muted);
            font-size: 13px;
        }

        .run-btn {
            width: 100%;
            margin-top: 8px;
            padding: 14px 18px;
            border: 0;
            border-radius: 12px;
            background: linear-gradient(180deg, var(--navy-2) 0%, var(--navy) 100%);
            color: #fff;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 22px rgba(11, 22, 51, 0.18);
            transition: transform .05s ease, opacity .15s ease, box-shadow .15s ease;
        }

        .run-btn:hover { opacity: 0.97; }
        .run-btn:active { transform: translateY(1px); }
        .run-btn:disabled {
            opacity: 0.58;
            cursor: not-allowed;
            box-shadow: none;
        }

        .summary-list {
            display: grid;
            gap: 12px;
        }

        .summary-item {
            padding: 14px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        .summary-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 5px;
        }

        .summary-value {
            font-size: 15px;
            font-weight: 700;
            line-height: 1.4;
            overflow-wrap: anywhere;
        }

        .summary-sub {
            margin-top: 6px;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }

        .alert {
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 18px;
            border: 1px solid transparent;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 14px;
        }

        .alert.error {
            color: var(--red-text);
            background: var(--red-bg);
            border-color: var(--red-border);
        }

        .alert.info {
            color: var(--blue-text);
            background: var(--blue-bg);
            border-color: var(--blue-border);
        }

        .results {
            display: grid;
            gap: 18px;
            margin-top: 20px;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            gap: 14px;
            align-items: center;
            margin-bottom: 16px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .status-pill.good {
            background: var(--green-bg);
            color: var(--green-text);
            border-color: var(--green-border);
        }

        .status-pill.warn {
            background: var(--amber-bg);
            color: var(--amber-text);
            border-color: var(--amber-border);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }

        .metric {
            padding: 14px;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--surface-2);
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            font-size: 18px;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .prediction-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 800;
            border: 1px solid transparent;
        }

        .prediction-chip.fake {
            background: var(--red-bg);
            color: var(--red-text);
            border-color: var(--red-border);
        }

        .prediction-chip.real {
            background: var(--green-bg);
            color: var(--green-text);
            border-color: var(--green-border);
        }

        .details-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 16px;
        }

        .detail-card {
            padding: 16px;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--surface-2);
        }

        .detail-title {
            margin: 0 0 12px;
            font-size: 14px;
            font-weight: 800;
        }

        .kv-list {
            display: grid;
            gap: 10px;
        }

        .kv-row {
            display: grid;
            grid-template-columns: 150px minmax(0, 1fr);
            gap: 10px;
            align-items: start;
        }

        .kv-key {
            color: var(--muted);
            font-size: 13px;
        }

        .kv-value {
            font-size: 13px;
            font-weight: 700;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .media-grid {
            display: grid;
            gap: 16px;
        }

        .video-wrap video {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: #000;
            max-height: 420px;
        }

        .thumb-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }

        .thumb {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            background: #fff;
        }

        .thumb img {
            display: block;
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            background: #e5e7eb;
        }

        .thumb-caption {
            padding: 8px;
            font-size: 12px;
            color: var(--muted);
            text-align: center;
            border-top: 1px solid var(--border);
            background: var(--surface-2);
        }

        .slider-wrap {
            margin-top: 14px;
            padding: 14px;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--surface-2);
        }

        .slider-head {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            margin-bottom: 8px;
        }

        .slider-value {
            font-weight: 700;
            font-size: 13px;
            color: var(--text);
        }

        .single-frame-preview {
            margin-top: 12px;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            background: #fff;
        }

        .single-frame-preview img {
            width: 100%;
            display: block;
            max-height: 420px;
            object-fit: contain;
            background: #0b1020;
        }

        details {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: #fff;
            overflow: hidden;
        }

        summary {
            cursor: pointer;
            list-style: none;
            padding: 15px 16px;
            font-weight: 700;
            background: var(--surface-2);
        }

        summary::-webkit-details-marker { display: none; }

        pre {
            margin: 0;
            padding: 16px;
            background: #07132e;
            color: #e7eefc;
            overflow-x: auto;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 13px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }

        .path-box {
            padding: 14px;
            border-radius: 14px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 13px;
            color: #344054;
            overflow-wrap: anywhere;
        }

        .micro-note {
            margin-top: 10px;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.5;
        }

        .tiny {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.45;
        }

        @media (max-width: 980px) {
            .layout { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: 1fr 1fr; }
            .details-grid { grid-template-columns: 1fr; }
            .hero { flex-direction: column; }
            .hero-right { width: 100%; justify-content: space-between; }
        }

        @media (max-width: 720px) {
            .grid-2 { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: 1fr; }
            .thumb-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .kv-row { grid-template-columns: 1fr; }
        }
    </style>

    <script>
        function toggleMode() {
            const mode = document.getElementById('input_mode').value;
            document.getElementById('video_block').style.display = (mode === 'video') ? 'block' : 'none';
            document.getElementById('frames_block').style.display = (mode === 'frames_dir') ? 'block' : 'none';
            updateSessionSummary();
        }

        function bindUploadInfo() {
            const videoInput = document.getElementById('video_file');
            const framesInput = document.getElementById('frames_files');
            const videoMeta = document.getElementById('video_meta');
            const framesMeta = document.getElementById('frames_meta');
            const clientVideo = document.getElementById('client_video_preview');
            const clientFrames = document.getElementById('client_frames_preview');
            const clientFramesGrid = document.getElementById('client_frames_grid');

            const selectedText = "{{ tr.selected }}";
            const selectedFilesText = "{{ tr.selected_files }}";
            const noFileText = "{{ tr.no_file_selected }}";
            const noFramesText = "{{ tr.no_frames_selected }}";

            if (videoInput) {
                videoInput.addEventListener('change', function() {
                    if (!videoInput.files || videoInput.files.length === 0) {
                        videoMeta.textContent = noFileText;
                        if (clientVideo) clientVideo.style.display = 'none';
                        updateSessionSummary();
                        return;
                    }

                    const f = videoInput.files[0];
                    videoMeta.textContent = selectedText + ': ' + f.name + ' · ' + formatBytes(f.size);

                    if (clientVideo) {
                        clientVideo.src = URL.createObjectURL(f);
                        clientVideo.style.display = 'block';
                    }

                    updateSessionSummary();
                });
            }

            if (framesInput) {
                framesInput.addEventListener('change', function() {
                    const files = framesInput.files ? Array.from(framesInput.files) : [];
                    const count = files.length;

                    if (count === 0) {
                        framesMeta.textContent = noFramesText;
                        if (clientFrames) clientFrames.style.display = 'none';
                        if (clientFramesGrid) clientFramesGrid.innerHTML = '';
                        updateSessionSummary();
                        return;
                    }

                    framesMeta.textContent = selectedFilesText + ': ' + count;

                    if (clientFrames && clientFramesGrid) {
                        clientFrames.style.display = 'block';
                        clientFramesGrid.innerHTML = '';

                        files
                            .filter(f => /\\.(jpg|jpeg|png|bmp)$/i.test(f.name))
                            .slice(0, 8)
                            .forEach((f, idx) => {
                                const item = document.createElement('div');
                                item.className = 'thumb';

                                const img = document.createElement('img');
                                img.src = URL.createObjectURL(f);

                                const cap = document.createElement('div');
                                cap.className = 'thumb-caption';
                                cap.textContent = f.name || ('frame ' + idx);

                                item.appendChild(img);
                                item.appendChild(cap);
                                clientFramesGrid.appendChild(item);
                            });
                    }

                    updateSessionSummary();
                });
            }
        }

        function updateSessionSummary() {
            const modelSel = document.getElementById('checkpoint_path');
            const modeSel = document.getElementById('input_mode');
            const deviceSel = document.getElementById('device_name');
            const videoInput = document.getElementById('video_file');
            const framesInput = document.getElementById('frames_files');

            const modelOut = document.getElementById('session_model');
            const modeOut = document.getElementById('session_mode');
            const deviceOut = document.getElementById('session_device');
            const sourceOut = document.getElementById('session_source');

            if (modelSel && modelOut) {
                modelOut.textContent = modelSel.options[modelSel.selectedIndex]?.text || '{{ tr.not_selected }}';
            }
            if (modeSel && modeOut) {
                modeOut.textContent = modeSel.value || '{{ tr.not_selected }}';
            }
            if (deviceSel && deviceOut) {
                deviceOut.textContent = deviceSel.value || '{{ tr.not_selected }}';
            }
            if (sourceOut) {
                if (modeSel.value === 'video') {
                    sourceOut.textContent = (videoInput.files && videoInput.files.length > 0)
                        ? videoInput.files[0].name
                        : '{{ tr.no_file_selected }}';
                } else {
                    sourceOut.textContent = (framesInput.files && framesInput.files.length > 0)
                        ? ('{{ tr.frames_selected }}: ' + framesInput.files.length)
                        : '{{ tr.not_selected }}';
                }
            }
        }

        function bindFormState() {
            const form = document.getElementById('infer_form');
            const btn = document.getElementById('run_btn');
            if (!form || !btn) return;

            form.addEventListener('submit', function() {
                btn.disabled = true;
                btn.textContent = '{{ tr.running_button }}';
            });
        }

        function formatBytes(bytes) {
            if (bytes < 1024) return bytes + ' B';
            const kb = bytes / 1024;
            if (kb < 1024) return kb.toFixed(1) + ' KB';
            const mb = kb / 1024;
            if (mb < 1024) return mb.toFixed(1) + ' MB';
            return (mb / 1024).toFixed(2) + ' GB';
        }

        function bindSampledFrameSlider() {
            const slider = document.getElementById('sampled_frame_slider');
            const target = document.getElementById('sampled_frame_target');
            const label = document.getElementById('sampled_frame_value');
            if (!slider || !target || !label) return;

            function updateFrame() {
                const idx = parseInt(slider.value, 10);
                const url = slider.getAttribute('data-frame-' + idx);
                const caption = slider.getAttribute('data-caption-' + idx) || ('Frame ' + idx);
                if (url) {
                    target.src = url;
                    label.textContent = caption;
                }
            }

            slider.addEventListener('input', updateFrame);
            updateFrame();
        }

        window.addEventListener('DOMContentLoaded', function() {
            toggleMode();
            bindUploadInfo();
            bindFormState();
            updateSessionSummary();
            bindSampledFrameSlider();

            const modelSel = document.getElementById('checkpoint_path');
            const modeSel = document.getElementById('input_mode');
            const deviceSel = document.getElementById('device_name');

            if (modelSel) modelSel.addEventListener('change', updateSessionSummary);
            if (modeSel) modeSel.addEventListener('change', updateSessionSummary);
            if (deviceSel) deviceSel.addEventListener('change', updateSessionSummary);
        });
    </script>
</head>
<body>
    <div class="page">
        <div class="hero">
            <div>
                <h1 class="title">{{ tr.page_title }}</h1>
                <p class="subtitle">{{ tr.hero_subtitle }}</p>
            </div>
            <div class="hero-right">
                <div class="lang-switch">
                    <a class="lang-link {% if lang == 'ru' %}active{% endif %}" href="{{ lang_ru_url }}">{{ tr.lang_ru }}</a>
                    <a class="lang-link {% if lang == 'en' %}active{% endif %}" href="{{ lang_en_url }}">{{ tr.lang_en }}</a>
                </div>
                <div class="hero-badge">{{ tr.hero_badge }}</div>
            </div>
        </div>

        {% if info_note %}
            <div class="alert info">{{ info_note }}</div>
        {% endif %}

        {% if error %}
            <div class="alert error">{{ error }}</div>
        {% endif %}

        <div class="layout">
            <div class="card">
                <div class="card-inner">
                    <h2 class="section-title">{{ tr.run_inference }}</h2>
                    <p class="section-subtitle">{{ tr.run_subtitle }}</p>

                    <form id="infer_form" method="post" enctype="multipart/form-data">
                        <input type="hidden" name="lang" value="{{ lang }}">

                        <div class="field">
                            <label for="checkpoint_path">{{ tr.model }}</label>
                            <select id="checkpoint_path" name="checkpoint_path" required>
                                {% if model_options %}
                                    {% for model in model_options %}
                                        <option value="{{ model.path }}" {% if model.path == selected_model %}selected{% endif %}>{{ model.label }}</option>
                                    {% endfor %}
                                {% else %}
                                    <option value="">{{ tr.no_models_found }}</option>
                                {% endif %}
                            </select>
                            <div class="hint">{{ tr.model_hint }}</div>
                        </div>

                        <div class="grid-2">
                            <div class="field">
                                <label for="input_mode">{{ tr.input_mode }}</label>
                                <select id="input_mode" name="input_mode" onchange="toggleMode()">
                                    <option value="video" {% if input_mode == "video" %}selected{% endif %}>{{ tr.video_label }}</option>
                                    <option value="frames_dir" {% if input_mode == "frames_dir" %}selected{% endif %}>{{ tr.frames_dir_label }}</option>
                                </select>
                                <div class="hint">{{ tr.input_mode_hint }}</div>
                            </div>

                            <div class="field">
                                <label for="device_name">{{ tr.device }}</label>
                                <select id="device_name" name="device_name">
                                    <option value="auto" {% if device_name == "auto" %}selected{% endif %}>auto</option>
                                    <option value="cpu" {% if device_name == "cpu" %}selected{% endif %}>cpu</option>
                                    <option value="mps" {% if device_name == "mps" %}selected{% endif %}>mps</option>
                                    <option value="cuda" {% if device_name == "cuda" %}selected{% endif %}>cuda</option>
                                </select>
                                <div class="hint">{{ tr.device_hint }}</div>
                            </div>
                        </div>

                        <div class="field">
                            <label>{{ tr.clip_length }}</label>
                            <input type="text" value="{{ fixed_frames_text }}" readonly>
                            <div class="hint">{{ tr.clip_length_hint }}</div>
                        </div>

                        <div id="video_block" class="field upload-box">
                            <label for="video_file">{{ tr.upload_video }}</label>
                            <input id="video_file" type="file" name="video_file" accept=".mp4,.avi,.mov,.mkv,.webm,.m4v,video/*">
                            <div class="hint">{{ tr.upload_video_hint }}</div>
                            <div class="upload-meta" id="video_meta">{{ tr.no_file_selected }}</div>

                            <div class="media-grid" style="margin-top:12px;">
                                <div class="video-wrap">
                                    <video id="client_video_preview" controls style="display:none;"></video>
                                </div>
                            </div>
                        </div>

                        <div id="frames_block" class="field upload-box">
                            <label for="frames_files">{{ tr.upload_frames }}</label>
                            <input id="frames_files" type="file" name="frames_files" webkitdirectory directory multiple accept=".jpg,.jpeg,.png,.bmp,image/*">
                            <div class="hint">{{ tr.upload_frames_hint }}</div>
                            <div class="upload-meta" id="frames_meta">{{ tr.no_frames_selected }}</div>

                            <div id="client_frames_preview" style="display:none; margin-top:12px;">
                                <div class="thumb-grid" id="client_frames_grid"></div>
                            </div>
                        </div>

                        <button id="run_btn" class="run-btn" type="submit" {% if not model_options %}disabled{% endif %}>{{ tr.run_button }}</button>
                    </form>
                </div>
            </div>

            <div class="card">
                <div class="card-inner">
                    <h2 class="section-title">{{ tr.session_summary }}</h2>
                    <p class="section-subtitle">{{ tr.session_subtitle }}</p>

                    <div class="summary-list">
                        <div class="summary-item">
                            <div class="summary-label">{{ tr.selected_model }}</div>
                            <div class="summary-value" id="session_model">{{ selected_model_label or tr.not_selected }}</div>
                            <div class="summary-sub">{{ tr.selected_model_sub }}</div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">{{ tr.selected_input_mode }}</div>
                            <div class="summary-value" id="session_mode">{{ input_mode }}</div>
                            <div class="summary-sub">{{ tr.selected_input_mode_sub }}</div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">{{ tr.requested_device }}</div>
                            <div class="summary-value" id="session_device">{{ device_name }}</div>
                            <div class="summary-sub">{{ tr.requested_device_sub }}</div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">{{ tr.current_source }}</div>
                            <div class="summary-value" id="session_source">{{ tr.no_file_selected }}</div>
                            <div class="summary-sub">{{ tr.current_source_sub }}</div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-label">{{ tr.important_limits }}</div>
                            <div class="summary-value">{{ tr.important_limits_value }}</div>
                            <div class="summary-sub">{{ tr.important_limits_sub }}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        {% if result_payload %}
            <div class="results">
                <div class="card">
                    <div class="card-inner">
                        <div class="result-header">
                            <div>
                                <h2 class="section-title">{{ tr.inference_result }}</h2>
                                <p class="section-subtitle" style="margin-bottom:0;">{{ tr.result_subtitle }}</p>
                            </div>

                            {% if result_view.fallback_used %}
                                <div class="status-pill warn">{{ tr.completed_with_fallback }}</div>
                            {% else %}
                                <div class="status-pill good">{{ tr.completed_successfully }}</div>
                            {% endif %}
                        </div>

                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">{{ tr.prediction }}</div>
                                <div class="metric-value">
                                    <span class="prediction-chip {{ result_view.prediction_class }}">
                                        {{ result_view.prediction_display }}
                                    </span>
                                </div>
                            </div>

                            <div class="metric">
                                <div class="metric-label">{{ tr.probability_fake }}</div>
                                <div class="metric-value">{{ result_view.probability_fake }}</div>
                            </div>

                            <div class="metric">
                                <div class="metric-label">{{ tr.device_used }}</div>
                                <div class="metric-value">{{ result_view.device_final }}</div>
                            </div>

                            <div class="metric">
                                <div class="metric-label">{{ tr.frames_used }}</div>
                                <div class="metric-value">{{ result_view.used_frames }}</div>
                            </div>
                        </div>

                        <div class="details-grid">
                            <div class="detail-card">
                                <h3 class="detail-title">{{ tr.run_details }}</h3>
                                <div class="kv-list">
                                    <div class="kv-row"><div class="kv-key">{{ tr.model_type }}</div><div class="kv-value">{{ result_view.model_type }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.fusion_type }}</div><div class="kv-value">{{ result_view.fusion_type }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.dataset }}</div><div class="kv-value">{{ result_view.dataset_name }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.threshold }}</div><div class="kv-value">{{ result_view.threshold }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.requested_device }}</div><div class="kv-value">{{ result_view.device_requested }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.final_device }}</div><div class="kv-value">{{ result_view.device_final }}</div></div>
                                    {% if result_view.fallback_used %}
                                    <div class="kv-row"><div class="kv-key">{{ tr.fallback }}</div><div class="kv-value">{{ result_view.device_fallback_from }} → {{ result_view.device_final }}</div></div>
                                    {% endif %}
                                </div>
                            </div>

                            <div class="detail-card">
                                <h3 class="detail-title">{{ tr.input_details }}</h3>
                                <div class="kv-list">
                                    <div class="kv-row"><div class="kv-key">{{ tr.input_type }}</div><div class="kv-value">{{ result_view.input_type }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.source_file }}</div><div class="kv-value">{{ result_view.source_name }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.source_path }}</div><div class="kv-value">{{ result_view.source_path }}</div></div>
                                    <div class="kv-row"><div class="kv-key">{{ tr.frames_used }}</div><div class="kv-value">{{ result_view.used_frames }}</div></div>
                                    {% if result_view.source_frames is not none %}
                                    <div class="kv-row"><div class="kv-key">{{ tr.source_frames }}</div><div class="kv-value">{{ result_view.source_frames }}</div></div>
                                    {% endif %}
                                    {% if result_view.face_preprocessing %}
                                    <div class="kv-row"><div class="kv-key">{{ tr.preprocessing }}</div><div class="kv-value">{{ result_view.face_preprocessing }}</div></div>
                                    {% endif %}
                                </div>
                            </div>
                        </div>

                        {% if result_view.has_fusion_weights %}
                        <div class="detail-card" style="margin-bottom:16px;">
                            <h3 class="detail-title">{{ tr.fusion_weights }}</h3>
                            <div class="kv-list">
                                <div class="kv-row"><div class="kv-key">alpha_spatial</div><div class="kv-value">{{ result_view.alpha_spatial }}</div></div>
                                <div class="kv-row"><div class="kv-key">alpha_temporal</div><div class="kv-value">{{ result_view.alpha_temporal }}</div></div>
                            </div>
                        </div>
                        {% endif %}

                        {% if original_video_data_uri %}
                        <div class="detail-card" style="margin-bottom:16px;">
                            <h3 class="detail-title">{{ tr.tested_video }}</h3>
                            <div class="video-wrap">
                                <video controls src="{{ original_video_data_uri }}"></video>
                            </div>
                            <div class="micro-note">{{ tr.video_preview_note }}</div>
                        </div>
                        {% elif video_preview_note %}
                        <div class="detail-card" style="margin-bottom:16px;">
                            <h3 class="detail-title">{{ tr.tested_video }}</h3>
                            <div class="micro-note">{{ video_preview_note }}</div>
                        </div>
                        {% endif %}

                        {% if sampled_frames %}
                        <div class="detail-card" style="margin-bottom:16px;">
                            <h3 class="detail-title">{{ tr.frames_used_for_inference }}</h3>

                            <div class="thumb-grid">
                                {% for frame in sampled_frames %}
                                <div class="thumb">
                                    <img src="{{ frame.url }}" alt="sampled frame {{ loop.index0 }}">
                                    <div class="thumb-caption">{{ frame.caption }}</div>
                                </div>
                                {% endfor %}
                            </div>

                            <div class="slider-wrap">
                                <div class="slider-head">
                                    <div class="tiny">{{ tr.preview_selected_frame }}</div>
                                    <div class="slider-value" id="sampled_frame_value">Frame 1</div>
                                </div>

                                <input
                                    id="sampled_frame_slider"
                                    type="range"
                                    min="0"
                                    max="{{ sampled_frames|length - 1 }}"
                                    step="1"
                                    value="0"
                                    {% for frame in sampled_frames %}
                                    data-frame-{{ loop.index0 }}="{{ frame.url }}"
                                    data-caption-{{ loop.index0 }}="{{ frame.caption }}"
                                    {% endfor %}
                                >

                                <div class="single-frame-preview">
                                    <img id="sampled_frame_target" src="{{ sampled_frames[0].url }}" alt="selected sampled frame">
                                </div>
                            </div>
                        </div>
                        {% endif %}

                        <details>
                            <summary>{{ tr.raw_result_details }}</summary>
                            <pre>{{ result_text }}</pre>
                        </details>

                        {% if result_view.fallback_used and result_view.device_fallback_reason %}
                        <div class="micro-note">
                            {{ tr.fallback_reason }}: {{ result_view.device_fallback_reason }}
                        </div>
                        {% endif %}
                    </div>
                </div>

                <div class="card">
                    <div class="card-inner">
                        <h2 class="section-title">{{ tr.saved_json_path }}</h2>
                        <p class="section-subtitle">{{ tr.saved_json_subtitle }}</p>
                        <div class="path-box">{{ saved_json }}</div>
                    </div>
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


def get_lang() -> str:
    lang = request.values.get("lang", "ru").strip().lower()
    return lang if lang in LANGS else "ru"


def tr(lang: str) -> Dict[str, str]:
    return TRANSLATIONS[lang]


def short_model_label(exp_dir: str) -> str:
    exp = exp_dir.lower()

    if "dfdc02_full" in exp and "adaptive" in exp and "bs8" in exp:
        return "A1 Full Adaptive (DFDC02, bs8) [recommended]"
    if "dfdc02_full" in exp and "adaptive" in exp:
        return "A1 Full Adaptive (DFDC02)"
    if "dfdc02_spatial" in exp:
        return "A2 Spatial Only (DFDC02)"
    if "dfdc02_sequential" in exp:
        return "A4 Sequential (DFDC02)"
    if "dfdc02_temporal" in exp:
        return "A3 Temporal Only (DFDC02)"
    if "concat" in exp:
        return "Full Concat"
    if "gate" in exp:
        return "Full Gate"

    return exp_dir


def discover_checkpoints() -> List[dict]:
    found = []

    for root in MODEL_SEARCH_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("best_model.pt"):
            exp_dir = p.parent.name
            found.append(
                {
                    "path": str(p.resolve()),
                    "exp_dir": exp_dir,
                    "label": short_model_label(exp_dir),
                }
            )

    unique = {}
    for item in found:
        unique[item["path"]] = item
    found = list(unique.values())

    def score(item: dict) -> Tuple[int, int, int, float]:
        exp = item["exp_dir"].lower()
        path = Path(item["path"])
        s1 = 1 if ("full" in exp and "adaptive" in exp) else 0
        s2 = 1 if "dfdc02" in exp else 0
        s3 = 1 if "bs8" in exp else 0
        mtime = path.stat().st_mtime if path.exists() else 0.0
        return (s1, s2, s3, mtime)

    found.sort(key=score, reverse=True)
    return found


def default_checkpoint(model_options: List[dict]) -> str:
    return model_options[0]["path"] if model_options else ""


def get_selected_model_label(model_options: List[dict], selected_path: str) -> str:
    for model in model_options:
        if model["path"] == selected_path:
            return model["label"]
    return "Not selected"


def format_result_text(payload: dict) -> str:
    result = payload["result"]
    lines = [
        f"Model type: {payload['model_type']}",
        f"Fusion type: {payload['fusion_type']}",
        f"Dataset name: {payload['dataset_name']}",
        f"Device: {payload['device']}",
    ]

    if payload.get("device_fallback_used"):
        lines.append(f"Device fallback: {payload['device_fallback_from']} -> {payload['device']}")

    lines += [
        "",
        f"Prediction: {result['pred_label']}",
        f"Probability fake: {result['probability_fake']:.6f}",
        f"Threshold: {result['decision_threshold']:.4f}",
    ]

    if "fusion_weights" in result:
        fw = result["fusion_weights"]
        lines += [
            "",
            "Fusion weights:",
            f"  alpha_spatial : {fw['alpha_spatial']:.6f}",
            f"  alpha_temporal: {fw['alpha_temporal']:.6f}",
        ]

    inp = payload["input"]
    lines += [
        "",
        "Input:",
        f"  type: {inp['input_type']}",
        f"  source: {inp['source_path']}",
        f"  used frames: {inp['num_used_frames']}",
    ]

    if "num_source_frames" in inp:
        lines.append(f"  source frames: {inp['num_source_frames']}")

    if "face_preprocessing" in inp:
        lines.append(f"  preprocessing: {inp['face_preprocessing']}")

    return "\n".join(lines)


def pil_image_to_data_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def rgb_frame_to_data_uri(frame_rgb) -> str:
    img = Image.fromarray(frame_rgb)
    return pil_image_to_data_uri(img, fmt="JPEG", quality=88)


def file_to_video_data_uri(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > INLINE_VIDEO_MAX_MB:
        return None

    mime = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".m4v": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(path.suffix.lower(), "video/mp4")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_result_view(payload: dict, requested_device: str, lang: str) -> Dict:
    result = payload["result"]
    inp = payload["input"]
    prediction = result["pred_label"]
    fusion = result.get("fusion_weights", {})
    tt = tr(lang)

    return {
        "prediction_display": tt["fake_detected"] if prediction == "fake" else tt["likely_real"],
        "prediction_class": "fake" if prediction == "fake" else "real",
        "probability_fake": f"{result['probability_fake']:.6f}",
        "threshold": f"{result['decision_threshold']:.4f}",
        "model_type": payload["model_type"],
        "fusion_type": payload["fusion_type"],
        "dataset_name": payload["dataset_name"],
        "device_requested": requested_device,
        "device_final": payload["device"],
        "fallback_used": payload.get("device_fallback_used", False),
        "device_fallback_from": payload.get("device_fallback_from"),
        "device_fallback_reason": payload.get("device_fallback_reason"),
        "input_type": inp["input_type"],
        "source_name": Path(inp["source_path"]).name,
        "source_path": inp["source_path"],
        "used_frames": inp["num_used_frames"],
        "source_frames": inp.get("num_source_frames"),
        "face_preprocessing": inp.get("face_preprocessing"),
        "has_fusion_weights": bool(fusion),
        "alpha_spatial": f"{fusion.get('alpha_spatial', 0.0):.6f}" if fusion else None,
        "alpha_temporal": f"{fusion.get('alpha_temporal', 0.0):.6f}" if fusion else None,
    }


def save_uploaded_video(video_storage, root_dir: Path, lang: str) -> Path:
    tt = tr(lang)

    if video_storage is None or not video_storage.filename:
        raise ValueError(tt["no_video_uploaded"])

    filename = secure_filename(video_storage.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"{tt['unsupported_video_format']}: {suffix}. "
            f"{tt['supported_video_formats']}: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    out_path = root_dir / filename
    video_storage.save(out_path)
    return out_path


def save_uploaded_frames(frames_list, root_dir: Path, lang: str) -> Path:
    tt = tr(lang)

    if not frames_list:
        raise ValueError(tt["no_frames_uploaded"])

    frames_root = root_dir / "frames_dir"
    frames_root.mkdir(parents=True, exist_ok=True)
    saved_any = False

    for fs in frames_list:
        if fs is None or not fs.filename:
            continue

        raw_name = fs.filename.replace("\\", "/")
        rel_path = Path(raw_name)

        if len(rel_path.parts) == 1:
            safe_rel = Path(secure_filename(rel_path.name))
        else:
            safe_parts = [secure_filename(part) for part in rel_path.parts if part not in ("", ".", "..")]
            safe_rel = Path(*safe_parts)

        if safe_rel.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        out_path = frames_root / safe_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fs.save(out_path)
        saved_any = True

    if not saved_any:
        raise ValueError(
            f"{tt['no_supported_images']} "
            f"{tt['supported_image_formats']}: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    image_files = [p for p in frames_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        raise ValueError(tt["no_frames_after_upload"])

    parent_counts = {}
    for p in image_files:
        parent_counts[p.parent] = parent_counts.get(p.parent, 0) + 1

    best_parent = max(parent_counts.items(), key=lambda x: x[1])[0]
    return best_parent


def is_mps_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    patterns = [
        "mps",
        "adaptive pool",
        "input sizes must be divisible by output sizes",
        "not implemented for mps",
        "placeholder storage has not been allocated",
    ]
    return any(p in text for p in patterns)


def build_preview_frames(
    input_path: Path,
    cfg,
    device,
    lang: str,
) -> List[dict]:
    tt = tr(lang)

    if input_path.is_dir():
        frames_rgb = prepare_frames_from_dir(input_path, cfg)
        base_caption = tt["sampled_frame"]
    else:
        frames_rgb = extract_face_frames_from_video(
            video_path=input_path,
            cfg=cfg,
            device=device,
        )
        base_caption = tt["used_face_crop"]

    frames_rgb = frames_rgb[:MAX_PREVIEW_FRAMES]
    items = []
    for i, frame_rgb in enumerate(frames_rgb):
        items.append(
            {
                "url": rgb_frame_to_data_uri(frame_rgb),
                "caption": f"{base_caption} {i + 1}",
            }
        )
    return items


def _run_app_inference_once(checkpoint_path_str: str, input_path: Path, device_name: str, lang: str):
    tt = tr(lang)
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{tt['checkpoint_not_found']}: {checkpoint_path}")

    device = get_device(device_name)

    checkpoint, cfg, model = load_checkpoint_and_cfg(
        checkpoint_path=checkpoint_path,
        device=device,
        override_device=device_name,
        use_amp=False,
    )

    spatial_tensor, temporal_tensor, prep_info = prepare_input_tensors(
        input_path=input_path,
        cfg=cfg,
        device=device,
    )

    result = run_inference(
        model=model,
        spatial_tensor=spatial_tensor,
        temporal_tensor=temporal_tensor,
        cfg=cfg,
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "dataset_name": cfg.dataset_name,
        "model_type": cfg.model_type,
        "fusion_type": cfg.fusion_type,
        "device": str(device),
        "input": prep_info,
        "result": result,
    }
    return payload, checkpoint_path, cfg, device


def run_app_inference(checkpoint_path_str: str, input_path: Path, device_name: str, lang: str):
    requested_device = device_name.strip().lower()

    try:
        payload, checkpoint_path, cfg, device = _run_app_inference_once(
            checkpoint_path_str=checkpoint_path_str,
            input_path=input_path,
            device_name=device_name,
            lang=lang,
        )
    except RuntimeError as exc:
        allow_fallback = requested_device in {"auto", "mps"}
        if not allow_fallback or not is_mps_runtime_error(exc):
            raise

        payload, checkpoint_path, cfg, device = _run_app_inference_once(
            checkpoint_path_str=checkpoint_path_str,
            input_path=input_path,
            device_name="cpu",
            lang=lang,
        )
        payload["device_fallback_used"] = True
        payload["device_fallback_from"] = requested_device
        payload["device_fallback_reason"] = str(exc)
    else:
        payload["device_fallback_used"] = False

    save_dir = checkpoint_path.parent
    safe_name = Path(payload["input"]["source_path"]).stem.replace(" ", "_")
    out_json = save_dir / f"app_infer_{safe_name}.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    sampled_frames = build_preview_frames(input_path=input_path, cfg=cfg, device=device, lang=lang)
    return payload, format_result_text(payload), str(out_json), cfg, sampled_frames


@app.errorhandler(413)
def too_large(_):
    lang = get_lang()
    tt = tr(lang)
    model_options = discover_checkpoints()
    selected_model = default_checkpoint(model_options)

    return render_template_string(
        HTML,
        tr=tt,
        lang=lang,
        lang_ru_url=f"/?lang=ru",
        lang_en_url=f"/?lang=en",
        error=tt["upload_too_large"],
        info_note=None,
        result_text=None,
        saved_json=None,
        result_payload=None,
        result_view=None,
        model_options=model_options,
        selected_model=selected_model,
        selected_model_label=get_selected_model_label(model_options, selected_model),
        input_mode="video",
        device_name="auto",
        fixed_frames_text=tt["clip_length_unknown"],
        original_video_data_uri=None,
        video_preview_note=None,
        sampled_frames=None,
    ), 413


@app.route("/", methods=["GET", "POST"])
def index():
    lang = get_lang()
    tt = tr(lang)

    error = None
    info_note = tt["info_note"]
    result_text = None
    saved_json = None
    result_payload = None
    result_view = None
    sampled_frames = None
    original_video_data_uri = None
    video_preview_note = None

    model_options = discover_checkpoints()
    selected_model = default_checkpoint(model_options)
    selected_model_label = get_selected_model_label(model_options, selected_model)
    input_mode = "video"
    device_name = "auto"
    fixed_frames_text = tt["clip_length_unknown"]

    if request.method == "POST":
        selected_model = request.form.get("checkpoint_path", "").strip()
        selected_model_label = get_selected_model_label(model_options, selected_model)
        input_mode = request.form.get("input_mode", "video").strip()
        device_name = request.form.get("device_name", "auto").strip()

        temp_root = Path(tempfile.mkdtemp(prefix="deepfake_mvp_"))

        try:
            if not selected_model:
                raise ValueError(tt["no_available_model"])

            checkpoint_path = Path(selected_model)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"{tt['checkpoint_not_found']}: {checkpoint_path}")

            preview_device = get_device("cpu")
            _, cfg_for_info, _ = load_checkpoint_and_cfg(
                checkpoint_path=checkpoint_path,
                device=preview_device,
                override_device="cpu",
                use_amp=False,
            )
            fixed_frames_text = f"{cfg_for_info.num_frames} frames (from checkpoint config)"

            if input_mode == "video":
                input_path = save_uploaded_video(request.files.get("video_file"), temp_root, lang)
                original_video_data_uri = file_to_video_data_uri(input_path)
                if original_video_data_uri is None:
                    size_mb = input_path.stat().st_size / (1024 * 1024)
                    if size_mb > INLINE_VIDEO_MAX_MB:
                        video_preview_note = f"{tt['video_preview_hidden']} ({size_mb:.1f} MB > {INLINE_VIDEO_MAX_MB} MB)."
            elif input_mode == "frames_dir":
                input_path = save_uploaded_frames(request.files.getlist("frames_files"), temp_root, lang)
            else:
                raise ValueError(f"{tt['unknown_input_mode']}: {input_mode}")

            result_payload, result_text, saved_json, cfg_used, sampled_frames = run_app_inference(
                checkpoint_path_str=selected_model,
                input_path=input_path,
                device_name=device_name,
                lang=lang,
            )
            result_view = build_result_view(result_payload, requested_device=device_name, lang=lang)
            fixed_frames_text = f"{cfg_used.num_frames} frames (from checkpoint config)"

        except Exception as e:
            msg = str(e)
            if "Одиночное изображение не подходит" in msg or "Single image input is not supported" in msg:
                error = tt["single_image_not_supported"]
            elif "Не удалось извлечь ни одного face crop" in msg or "No face crop" in msg:
                error = tt["no_face_detected"]
            elif "Не удалось открыть видео" in msg or "Не удалось определить число кадров" in msg:
                error = tt["invalid_video_decode"]
            else:
                error = msg
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    return render_template_string(
        HTML,
        tr=tt,
        lang=lang,
        lang_ru_url=f"/?lang=ru",
        lang_en_url=f"/?lang=en",
        error=error,
        info_note=info_note,
        result_text=result_text,
        saved_json=saved_json,
        result_payload=result_payload,
        result_view=result_view,
        model_options=model_options,
        selected_model=selected_model,
        selected_model_label=selected_model_label,
        input_mode=input_mode,
        device_name=device_name,
        fixed_frames_text=fixed_frames_text,
        original_video_data_uri=original_video_data_uri,
        video_preview_note=video_preview_note,
        sampled_frames=sampled_frames,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)