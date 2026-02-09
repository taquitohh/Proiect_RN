"""Flask UI for chair type classification and Blender script generation."""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blender_scripts.chair_generator import generate_chair_script
from src.blender_scripts.table_generator import generate_table_script
from src.blender_scripts.cabinet_generator import generate_cabinet_script


LABEL_MAPS = {
    "chair": {
        0: "Simple Chair",
        1: "Chair with Backrest",
        2: "Bar Chair",
        3: "Stool",
    },
    "table": {
        0: "Low Table",
        1: "Dining Table",
        2: "Bar Table",
    },
    "cabinet": {
        0: "Single Door",
        1: "Double Door",
        2: "Tall Cabinet",
    },
}

SCALER_PATHS = {
    "chair": os.path.join("config", "preprocessing_params.pkl"),
    "table": os.path.join("config", "table_scaler.pkl"),
    "cabinet": os.path.join("config", "cabinet_scaler.pkl"),
}
MODEL_PATHS = {
    "chair": os.path.join("models", "trained_model.h5"),
    "table": os.path.join("models", "table_model.h5"),
    "cabinet": os.path.join("models", "cabinet_model.h5"),
}
BLENDER_API_URL = os.environ.get("BLENDER_API_URL", "http://127.0.0.1:5001/render")


def load_scaler(path: str):
    """Load the preprocessing scaler from disk."""
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def load_model(path: str) -> keras.Model:
    """Load the trained Keras model from disk."""
    return keras.models.load_model(path)


ARTIFACT_CACHE: dict[str, tuple[keras.Model, object]] = {}


def load_artifacts(object_type: str) -> tuple[keras.Model, object]:
    """Load model and scaler on demand for a given object type."""
    if object_type in ARTIFACT_CACHE:
        return ARTIFACT_CACHE[object_type]

    scaler_path = SCALER_PATHS.get(object_type)
    model_path = MODEL_PATHS.get(object_type)
    if not scaler_path or not model_path:
        raise ValueError(f"Unknown object_type: {object_type}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Please run preprocessing first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")

    print(f"Loading model: {model_path}")
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)
    print("Model loaded successfully.")
    ARTIFACT_CACHE[object_type] = (model, scaler)
    return model, scaler


def check_blender_api(timeout: float = 1.2) -> tuple[bool, str | None]:
    request_obj = urllib.request.Request(BLENDER_API_URL, method="OPTIONS")
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            if response.status >= 400:
                return False, f"HTTP {response.status}"
        return True, None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return False, str(exc)


FEATURE_COLUMNS = {
    "chair": [
        "seat_height",
        "seat_width",
        "seat_depth",
        "leg_count",
        "leg_thickness",
        "has_backrest",
        "backrest_height",
        "style_variant",
    ],
    "table": [
        "table_height",
        "table_width",
        "table_depth",
        "leg_count",
        "leg_thickness",
        "has_apron",
        "style_variant",
    ],
    "cabinet": [
        "cabinet_height",
        "cabinet_width",
        "cabinet_depth",
        "wall_thickness",
        "door_type",
        "door_count",
        "style_variant",
    ],
}


def build_input_array(object_type: str, values: dict) -> pd.DataFrame:
    """Build a DataFrame with feature names to match scaler training."""
    if object_type == "table":
        row = {
            "table_height": values["table_height"],
            "table_width": values["table_width"],
            "table_depth": values["table_depth"],
            "leg_count": values["table_leg_count"],
            "leg_thickness": values["table_leg_thickness"],
            "has_apron": values["table_has_apron"],
            "style_variant": values["table_style_variant"],
        }
    elif object_type == "cabinet":
        row = {
            "cabinet_height": values["cabinet_height"],
            "cabinet_width": values["cabinet_width"],
            "cabinet_depth": values["cabinet_depth"],
            "wall_thickness": values["wall_thickness"],
            "door_type": values["door_type"],
            "door_count": values["door_count"],
            "style_variant": values["cabinet_style_variant"],
        }
    else:
        row = {
            "seat_height": values["seat_height"],
            "seat_width": values["seat_width"],
            "seat_depth": values["seat_depth"],
            "leg_count": values["leg_count"],
            "leg_thickness": values["leg_size"],
            "has_backrest": values["has_backrest"],
            "backrest_height": values["backrest_height"],
            "style_variant": values["style_variant"],
        }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS[object_type])


app = Flask(__name__)


@app.route("/status", methods=["GET"])
def status():
    blender_ok, blender_error = check_blender_api()
    chair_ok = os.path.exists(SCALER_PATHS["chair"]) and os.path.exists(MODEL_PATHS["chair"])
    return jsonify(
        {
            "ui_ok": True,
            "model_ok": chair_ok,
            "blender_api_ok": blender_ok,
            "blender_api_url": BLENDER_API_URL,
            "blender_api_error": blender_error,
        }
    )


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Chair Type Classifier (Etapa 5.4)</title>
        <style>
            :root {
                --bg: #0e1116;
                --ink: #f2f4f8;
                --muted: #9aa3af;
                --paper: #151a22;
                --accent: #4f8c7a;
                --accent-2: #c97a5a;
                --shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
                --border: rgba(255, 255, 255, 0.08);
            }

            * { box-sizing: border-box; }

            body {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                margin: 0;
                background: radial-gradient(1200px 520px at 18% -12%, rgba(79, 140, 122, 0.35) 0%, rgba(79, 140, 122, 0) 62%),
                    radial-gradient(980px 680px at 82% 115%, rgba(201, 122, 90, 0.32) 0%, rgba(201, 122, 90, 0) 60%),
                    linear-gradient(180deg, #0e1116 0%, #0b0f14 100%);
                color: var(--ink);
            }

            .page {
                max-width: 1100px;
                margin: 40px auto 64px;
                padding: 0 20px;
                animation: fadeIn 600ms ease-out;
            }

            .hero {
                display: grid;
                grid-template-columns: minmax(260px, 1.2fr) minmax(220px, 0.8fr);
                gap: 24px;
                align-items: end;
                margin-bottom: 24px;
            }

            .title {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: clamp(28px, 4vw, 40px);
                margin: 0 0 8px;
                letter-spacing: -0.02em;
            }

            .subtitle {
                margin: 0;
                max-width: 540px;
                color: var(--muted);
            }

            .badge {
                justify-self: end;
                padding: 10px 16px;
                border-radius: 999px;
                background: var(--accent);
                color: #fff;
                font-weight: 600;
                box-shadow: var(--shadow);
            }

            .status-board {
                justify-self: end;
                display: grid;
                gap: 8px;
                padding: 14px 16px;
                border-radius: 18px;
                background: rgba(21, 26, 34, 0.95);
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                min-width: 240px;
            }

            .status-title {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.2em;
                color: var(--muted);
            }

            .status-item {
                display: grid;
                grid-template-columns: 10px 1fr;
                gap: 10px;
                align-items: center;
            }

            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #6f7b8b;
                box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.04);
            }

            .status-dot.ok { background: #4fb48c; }
            .status-dot.down { background: #d26c6c; }
            .status-dot.pending { background: #d1a15f; }

            .status-text {
                display: grid;
                gap: 2px;
                font-size: 13px;
            }

            .status-label {
                font-weight: 600;
                color: var(--ink);
            }

            .status-meta {
                font-size: 12px;
                color: var(--muted);
            }

            .status-note {
                font-size: 12px;
                color: var(--muted);
                line-height: 1.4;
            }

            .card {
                background: var(--paper);
                border-radius: 20px;
                padding: 20px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border);
            }

            .layout {
                display: grid;
                grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr);
                gap: 22px;
                align-items: stretch;
            }

            .panel {
                display: flex;
                flex-direction: column;
                gap: 16px;
                height: 100%;
            }

            .input-panel .grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
            }

            .input-panel input,
            .input-panel select {
                padding: 8px 10px;
            }

            .panel-title {
                font-size: 14px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
                margin: 0 0 6px;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 14px;
            }

            label {
                font-weight: 600;
                display: block;
                margin-bottom: 6px;
            }

            input, select {
                width: 100%;
                padding: 9px 10px;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: #0f141b;
                color: var(--ink);
                font-size: 14px;
            }

            button {
                border: 0;
                background: var(--accent);
                color: #fff;
                padding: 10px 16px;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: var(--shadow);
                transition: transform 120ms ease, background 120ms ease;
            }

            button:hover { transform: translateY(-1px); background: #1f5d46; }

            .box {
                margin-top: 18px;
                padding: 16px;
                border-radius: 16px;
                background: #0f141b;
                border: 1px solid var(--border);
            }

            .result-title {
                font-size: clamp(20px, 3vw, 28px);
                margin: 0 0 8px;
                color: var(--ink);
            }

            .output-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 10px;
            }

            .script-wrap {
                position: relative;
                flex: 1;
                display: flex;
            }

            .copy-btn {
                position: absolute;
                top: 8px;
                right: 8px;
                background: rgba(15, 20, 27, 0.9);
                border: 1px solid var(--border);
                color: var(--ink);
                padding: 6px 8px;
                border-radius: 10px;
                box-shadow: var(--shadow);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 34px;
                height: 30px;
            }

            .copy-btn svg {
                width: 16px;
                height: 16px;
                fill: var(--ink);
            }

            pre {
                background: #0b0f14;
                color: #e6e9ef;
                padding: 14px;
                border-radius: 12px;
                overflow-x: auto;
                font-size: 13px;
                line-height: 1.5;
            }

            .code-block {
                max-height: 420px;
                overflow: auto;
                white-space: pre;
            }

            .script-wrap .code-block {
                padding-top: 36px;
                flex: 1;
                max-height: none;
                min-height: 260px;
            }

            .output-card {
                display: flex;
                flex-direction: column;
                gap: 16px;
                height: 100%;
            }

            .output-block {
                background: #0f141b;
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 14px;
            }

            .output-block h3 {
                margin: 0 0 10px;
                font-size: 18px;
            }

            .preview {
                margin-top: 14px;
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid var(--border);
                background: #0b0f14;
            }


            .preview img {
                width: 100%;
                display: block;
            }

            .preview-note {
                color: var(--muted);
                font-size: 12px;
                margin-top: 8px;
            }

            .scroll-top {
                position: fixed;
                right: 24px;
                bottom: 24px;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                border: 1px solid var(--border);
                background: rgba(15, 20, 27, 0.9);
                color: var(--ink);
                display: grid;
                place-items: center;
                cursor: pointer;
                box-shadow: var(--shadow);
                opacity: 0;
                pointer-events: none;
                transition: opacity 160ms ease, transform 160ms ease;
                transform: translateY(6px);
                z-index: 100;
            }

            .scroll-top.show {
                opacity: 1;
                pointer-events: auto;
                transform: translateY(0);
            }

            .section-title {
                font-weight: 700;
                margin: 10px 0 8px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 11px;
            }

            .hidden {
                display: none;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 720px) {
                .hero { grid-template-columns: 1fr; }
                .layout { grid-template-columns: 1fr; }
                .input-panel .grid { grid-template-columns: 1fr; }
            }

            @media (max-width: 1100px) {
                .input-panel .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
        </style>
    </head>
    <body>
        <div id="top"></div>
        <div class="page">
            <div class="hero">
                <div>
                    <h1 class="title">Chair/Table/Cabinet Classifier + Blender Preview</h1>
                    <p class="subtitle">Selectează obiectul, completează parametrii și apasă Predict.</p>
                </div>
            </div>

            <div class="layout">
                <div class="panel input-panel">
                    <div class="panel-title">Input</div>
                    <form method="post" class="card">
                        <div class="grid">
                            <div>
                                <label for="object_type">object_type</label>
                                <select name="object_type" id="object_type" onchange="updateObjectTypeUI()">
                                    <option value="chair" {% if values.object_type == "chair" %}selected{% endif %}>chair</option>
                                    <option value="table" {% if values.object_type == "table" %}selected{% endif %}>table</option>
                                    <option value="cabinet" {% if values.object_type == "cabinet" %}selected{% endif %}>cabinet</option>
                                </select>
                            </div>
                        </div>

                        <div class="section-title chair-only">Sezut</div>
                        <div class="grid chair-only">
                        <div>
                            <label for="seat_height">seat_height</label>
                            <input type="number" step="0.01" min="0.4" max="0.8" name="seat_height" id="seat_height" value="{{ values.seat_height }}" required />
                        </div>
                        <div>
                            <label for="seat_width">seat_width</label>
                            <input type="number" step="0.01" min="0.35" max="0.6" name="seat_width" id="seat_width" value="{{ values.seat_width }}" required />
                        </div>
                        <div>
                            <label for="seat_depth">seat_depth</label>
                            <input type="number" step="0.01" min="0.35" max="0.6" name="seat_depth" id="seat_depth" value="{{ values.seat_depth }}" required />
                        </div>
                        </div>

                        <div class="section-title chair-only">Picioare</div>
                        <div class="grid chair-only">
                            <div>
                                <label for="leg_count">leg_count</label>
                                <input type="number" step="1" min="3" max="5" name="leg_count" id="leg_count" value="{{ values.leg_count }}" required />
                            </div>
                            <div>
                                <label for="leg_shape">leg_shape</label>
                                <select name="leg_shape" id="leg_shape">
                                    <option value="square" {% if values.leg_shape == "square" %}selected{% endif %}>square</option>
                                    <option value="round" {% if values.leg_shape == "round" %}selected{% endif %}>round</option>
                                </select>
                            </div>
                            <div>
                                <label for="leg_size">leg_size</label>
                                <input type="number" step="0.01" min="0.03" max="0.08" name="leg_size" id="leg_size" value="{{ values.leg_size }}" required />
                            </div>
                        </div>

                        <div class="section-title chair-only">Spatar</div>
                        <div class="grid chair-only">
                            <div>
                                <label for="has_backrest">has_backrest</label>
                                <select name="has_backrest" id="has_backrest">
                                    <option value="0" {% if values.has_backrest == 0 %}selected{% endif %}>0</option>
                                    <option value="1" {% if values.has_backrest == 1 %}selected{% endif %}>1</option>
                                </select>
                            </div>
                            <div>
                                <label for="backrest_height">backrest_height</label>
                                <input type="number" step="0.01" min="0.2" max="0.5" name="backrest_height" id="backrest_height" value="{{ values.backrest_height }}" required />
                            </div>
                            <div>
                                <label for="style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="style_variant" id="style_variant" value="{{ values.style_variant }}" required />
                            </div>
                        </div>

                        <div class="section-title table-only">Masa</div>
                        <div class="grid table-only">
                            <div>
                                <label for="table_height">table_height</label>
                                <input type="number" step="0.01" min="0.35" max="0.9" name="table_height" id="table_height" value="{{ values.table_height }}" required />
                            </div>
                            <div>
                                <label for="table_width">table_width</label>
                                <input type="number" step="0.01" min="0.5" max="1.4" name="table_width" id="table_width" value="{{ values.table_width }}" required />
                            </div>
                            <div>
                                <label for="table_depth">table_depth</label>
                                <input type="number" step="0.01" min="0.5" max="1.0" name="table_depth" id="table_depth" value="{{ values.table_depth }}" required />
                            </div>
                            <div>
                                <label for="table_leg_count">leg_count</label>
                                <input type="number" step="1" min="3" max="4" name="table_leg_count" id="table_leg_count" value="{{ values.table_leg_count }}" required />
                            </div>
                            <div>
                                <label for="table_leg_thickness">leg_thickness</label>
                                <input type="number" step="0.01" min="0.04" max="0.09" name="table_leg_thickness" id="table_leg_thickness" value="{{ values.table_leg_thickness }}" required />
                            </div>
                            <div>
                                <label for="table_has_apron">has_apron</label>
                                <select name="table_has_apron" id="table_has_apron">
                                    <option value="0" {% if values.table_has_apron == 0 %}selected{% endif %}>0</option>
                                    <option value="1" {% if values.table_has_apron == 1 %}selected{% endif %}>1</option>
                                </select>
                            </div>
                            <div>
                                <label for="table_style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="table_style_variant" id="table_style_variant" value="{{ values.table_style_variant }}" required />
                            </div>
                        </div>

                        <div class="section-title cabinet-only">Dulap</div>
                        <div class="grid cabinet-only">
                            <div>
                                <label for="cabinet_height">cabinet_height</label>
                                <input type="number" step="0.01" min="1.0" max="2.2" name="cabinet_height" id="cabinet_height" value="{{ values.cabinet_height }}" required />
                            </div>
                            <div>
                                <label for="cabinet_width">cabinet_width</label>
                                <input type="number" step="0.01" min="0.6" max="1.6" name="cabinet_width" id="cabinet_width" value="{{ values.cabinet_width }}" required />
                            </div>
                            <div>
                                <label for="cabinet_depth">cabinet_depth</label>
                                <input type="number" step="0.01" min="0.3" max="0.8" name="cabinet_depth" id="cabinet_depth" value="{{ values.cabinet_depth }}" required />
                            </div>
                            <div>
                                <label for="wall_thickness">wall_thickness</label>
                                <input type="number" step="0.001" min="0.015" max="0.05" name="wall_thickness" id="wall_thickness" value="{{ values.wall_thickness }}" required />
                            </div>
                            <div>
                                <label for="door_type">door_type</label>
                                <select name="door_type" id="door_type">
                                    <option value="0" {% if values.door_type == 0 %}selected{% endif %}>flush_door</option>
                                    <option value="1" {% if values.door_type == 1 %}selected{% endif %}>inset_door</option>
                                </select>
                            </div>
                            <div>
                                <label for="door_count">door_count</label>
                                <input type="number" step="1" min="1" max="2" name="door_count" id="door_count" value="{{ values.door_count }}" required />
                            </div>
                            <div>
                                <label for="cabinet_style_variant">style_variant</label>
                                <input type="number" step="1" min="0" max="2" name="cabinet_style_variant" id="cabinet_style_variant" value="{{ values.cabinet_style_variant }}" required />
                            </div>
                        </div>

                        <div style="margin-top: 12px;">
                            <button type="submit">Predict</button>
                        </div>
                    </form>
                    <div class="status-board">
                        <div class="status-title">Status pornire</div>
                        <div class="status-item">
                            <span class="status-dot ok"></span>
                            <div class="status-text">
                                <div class="status-label">UI Flask</div>
                                <div class="status-meta">pornit</div>
                            </div>
                        </div>
                        <div class="status-item">
                            <span class="status-dot ok"></span>
                            <div class="status-text">
                                <div class="status-label">Model RN</div>
                                <div class="status-meta">incarcat</div>
                            </div>
                        </div>
                        <div class="status-item">
                            <span class="status-dot pending" id="status-blender-dot"></span>
                            <div class="status-text">
                                <div class="status-label">Blender API</div>
                                <div class="status-meta" id="status-blender-meta">verificare...</div>
                            </div>
                        </div>
                        <div class="status-note">Pentru preview randat, porneste Blender API.</div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-title">Output</div>
                    <div class="card output-card">
                        <div class="output-block">
                            <h3>Blender preview</h3>
                            <div class="preview" id="preview-container">
                                <img
                                    id="preview-image"
                                    src="{% if result and result.preview_src %}{{ result.preview_src }}{% endif %}"
                                    alt="Blender preview"
                                    style="{% if not (result and result.preview_src) %}display: none;{% endif %}"
                                />
                            </div>
                            <div class="preview-note" id="preview-note">
                                {% if result and result.preview_error %}
                                    Preview error: {{ result.preview_error }}
                                {% else %}
                                    Blender API pe 127.0.0.1:5001.
                                {% endif %}
                            </div>
                        </div>

                        <div class="output-block">
                            {% if result %}
                                <div class="result-title">Predicted variant: {{ result.label }}</div>
                                <div><strong>Confidence:</strong> {{ "%.2f" | format(result.confidence) }}</div>
                                <div><strong>Probabilities:</strong></div>
                                <pre class="code-block">{{ result.probabilities | tojson(indent=2) }}</pre>
                            {% else %}
                                <div class="result-title">Prediction</div>
                                <div class="preview-note">Completează datele și apasă Predict.</div>
                            {% endif %}
                        </div>

                        <div class="output-block" style="display: flex; flex-direction: column; gap: 10px; flex: 1;">
                            <div class="output-header">
                                <div><strong>Generated Blender script:</strong></div>
                            </div>
                            <div class="script-wrap">
                                <button class="copy-btn" type="button" onclick="copyScript()" aria-label="Copy script">
                                    <svg viewBox="0 0 24 24" role="img" aria-hidden="true">
                                        <path d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H10V7h9v14z"/>
                                    </svg>
                                </button>
                                <pre id="blender-script" class="code-block">{% if result %}{{ result.script }}{% endif %}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <button class="scroll-top" type="button" id="scroll-top" aria-label="Scroll to top">
                ↑
            </button>

        <script>
            const hasBackrest = document.getElementById('has_backrest');
            const backrestHeight = document.getElementById('backrest_height');
            const objectTypeSelect = document.getElementById('object_type');
            const scrollTopButton = document.getElementById('scroll-top');
            const blenderDot = document.getElementById('status-blender-dot');
            const blenderMeta = document.getElementById('status-blender-meta');

            function updateBlenderStatus(status) {
                if (!blenderDot || !blenderMeta) {
                    return;
                }
                if (status && status.blender_api_ok) {
                    blenderDot.classList.remove('pending', 'down');
                    blenderDot.classList.add('ok');
                    blenderMeta.textContent = 'conectat';
                } else {
                    blenderDot.classList.remove('pending', 'ok');
                    blenderDot.classList.add('down');
                    blenderMeta.textContent = 'nepornit';
                }
            }

            function syncBackrest() {
                if (!hasBackrest || !backrestHeight) {
                    return;
                }
                if (objectTypeSelect && objectTypeSelect.value !== 'chair') {
                    backrestHeight.setAttribute('readonly', 'readonly');
                    return;
                }
                if (hasBackrest.value === '0') {
                    backrestHeight.value = '0.0';
                    backrestHeight.min = '0.0';
                    backrestHeight.max = '0.0';
                    backrestHeight.setAttribute('readonly', 'readonly');
                } else {
                    backrestHeight.min = '0.2';
                    backrestHeight.max = '0.5';
                    if (parseFloat(backrestHeight.value) < 0.2) {
                        backrestHeight.value = '0.2';
                    }
                    backrestHeight.removeAttribute('readonly');
                }
            }

            if (hasBackrest) {
                hasBackrest.addEventListener('change', syncBackrest);
            }

            function updateObjectTypeUI() {
                const chairBlocks = document.querySelectorAll('.chair-only');
                const tableBlocks = document.querySelectorAll('.table-only');
                const cabinetBlocks = document.querySelectorAll('.cabinet-only');
                const objectType = objectTypeSelect ? objectTypeSelect.value : 'chair';

                chairBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'chair'));
                tableBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'table'));
                cabinetBlocks.forEach((node) => node.classList.toggle('hidden', objectType !== 'cabinet'));

                const toggleInputs = (blocks, enabled) => {
                    blocks.forEach((node) => {
                        node.querySelectorAll('input, select').forEach((field) => {
                            if (enabled) {
                                field.removeAttribute('disabled');
                                if (field.dataset.wasRequired === 'true') {
                                    field.setAttribute('required', 'required');
                                    delete field.dataset.wasRequired;
                                }
                            } else {
                                field.setAttribute('disabled', 'disabled');
                                if (field.hasAttribute('required')) {
                                    field.dataset.wasRequired = 'true';
                                    field.removeAttribute('required');
                                }
                            }
                        });
                    });
                };

                toggleInputs(chairBlocks, objectType === 'chair');
                toggleInputs(tableBlocks, objectType === 'table');
                toggleInputs(cabinetBlocks, objectType === 'cabinet');

                syncBackrest();
            }

            if (objectTypeSelect) {
                objectTypeSelect.addEventListener('change', updateObjectTypeUI);
            }
            updateObjectTypeUI();

            function copyScript() {
                const script = document.getElementById('blender-script');
                if (!script) {
                    return;
                }
                navigator.clipboard.writeText(script.textContent || '')
                    .then(() => alert('Script copied to clipboard.'))
                    .catch(() => alert('Copy failed.'));
            }


            function updateScrollTopButton() {
                if (!scrollTopButton) {
                    return;
                }
                const scrollPosition = window.pageYOffset
                    || document.documentElement.scrollTop
                    || document.body.scrollTop
                    || 0;
                if (scrollPosition > 220) {
                    scrollTopButton.classList.add('show');
                } else {
                    scrollTopButton.classList.remove('show');
                }
            }

            function scrollToTopImmediate() {
                const topAnchor = document.getElementById('top');
                if (topAnchor) {
                    topAnchor.scrollIntoView({ behavior: 'auto' });
                }
                window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
                document.documentElement.scrollTop = 0;
                document.body.scrollTop = 0;
            }

            if ('scrollRestoration' in history) {
                history.scrollRestoration = 'manual';
            }

            window.addEventListener('scroll', updateScrollTopButton, { passive: true });
            window.addEventListener('load', () => {
                scrollToTopImmediate();
                setTimeout(scrollToTopImmediate, 0);
                updateScrollTopButton();
                fetch('/status')
                    .then((response) => response.json())
                    .then((data) => updateBlenderStatus(data))
                    .catch(() => updateBlenderStatus({ blender_api_ok: false }));
            });
            window.addEventListener('pageshow', (event) => {
                if (event.persisted) {
                    scrollToTopImmediate();
                }
            });

            window.addEventListener('beforeunload', () => {
                document.documentElement.scrollTop = 0;
                document.body.scrollTop = 0;
            });

            if (scrollTopButton) {
                scrollTopButton.addEventListener('click', () => {
                    window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
                });
            }
        </script>
        </div>
    </body>
</html>
"""


def parse_float(name: str, default: float) -> float:
        value = request.form.get(name, default)
        return float(value)


def parse_int(name: str, default: int) -> int:
        value = request.form.get(name, default)
        return int(value)


def parse_str(name: str, default: str) -> str:
    value = request.form.get(name, default)
    return str(value)


def request_preview(payload: dict) -> tuple[str | None, str | None]:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            BLENDER_API_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body.get("image_base64"), None
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
            details = body.get("details")
            if details:
                return None, f"{exc}: {details}"
            return None, f"{exc}: {body.get('error', 'Unknown error')}"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, str(exc)
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
        values = {
                "object_type": "chair",
                "seat_height": 0.55,
                "seat_width": 0.45,
                "seat_depth": 0.45,
                "leg_count": 4,
            "leg_shape": "square",
            "leg_size": 0.05,
                "has_backrest": 1,
                "backrest_height": 0.25,
                "style_variant": 0,
            "table_height": 0.75,
            "table_width": 1.0,
            "table_depth": 0.7,
            "table_leg_count": 4,
            "table_leg_thickness": 0.06,
            "table_has_apron": 1,
            "table_style_variant": 0,
            "cabinet_height": 1.8,
            "cabinet_width": 1.0,
            "cabinet_depth": 0.5,
            "wall_thickness": 0.03,
            "door_type": 0,
            "door_count": 2,
            "cabinet_style_variant": 0,
            "rotate_yaw": 35.0,
            "rotate_pitch": 15.0,
        }

        result = None
        if request.method == "POST":
                values["object_type"] = parse_str("object_type", values["object_type"])
                object_type = values["object_type"] if values["object_type"] in LABEL_MAPS else "chair"
                values["object_type"] = object_type

                if object_type == "table":
                    values["table_height"] = parse_float("table_height", values["table_height"])
                    values["table_width"] = parse_float("table_width", values["table_width"])
                    values["table_depth"] = parse_float("table_depth", values["table_depth"])
                    values["table_leg_count"] = parse_int("table_leg_count", values["table_leg_count"])
                    values["table_leg_thickness"] = parse_float("table_leg_thickness", values["table_leg_thickness"])
                    values["table_has_apron"] = parse_int("table_has_apron", values["table_has_apron"])
                    values["table_style_variant"] = parse_int("table_style_variant", values["table_style_variant"])

                    model, scaler = load_artifacts("table")
                    features = build_input_array("table", values)
                    scaled_features = scaler.transform(features)
                    probabilities = model.predict(scaled_features, verbose=0)[0]
                    predicted_label = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))

                    script = generate_table_script(
                        table_height=values["table_height"],
                        table_width=values["table_width"],
                        table_depth=values["table_depth"],
                        leg_count=values["table_leg_count"],
                        leg_thickness=values["table_leg_thickness"],
                        has_apron=values["table_has_apron"],
                        style_variant=values["table_style_variant"],
                    )

                    preview_payload = {
                        "object_type": "table",
                        "table_height": values["table_height"],
                        "table_width": values["table_width"],
                        "table_depth": values["table_depth"],
                        "leg_count": values["table_leg_count"],
                        "leg_thickness": values["table_leg_thickness"],
                        "has_apron": values["table_has_apron"],
                        "style_variant": values["table_style_variant"],
                        "rotate_yaw": values["rotate_yaw"],
                        "rotate_pitch": values["rotate_pitch"],
                    }
                    preview_image, preview_error = request_preview(preview_payload)
                    preview_src = None
                    if preview_image:
                        preview_src = f"data:image/png;base64,{preview_image}"
                    label_map = LABEL_MAPS["table"]
                elif object_type == "cabinet":
                    values["cabinet_height"] = parse_float("cabinet_height", values["cabinet_height"])
                    values["cabinet_width"] = parse_float("cabinet_width", values["cabinet_width"])
                    values["cabinet_depth"] = parse_float("cabinet_depth", values["cabinet_depth"])
                    values["wall_thickness"] = parse_float("wall_thickness", values["wall_thickness"])
                    values["door_type"] = parse_int("door_type", values["door_type"])
                    values["door_count"] = parse_int("door_count", values["door_count"])
                    values["cabinet_style_variant"] = parse_int("cabinet_style_variant", values["cabinet_style_variant"])

                    model, scaler = load_artifacts("cabinet")
                    features = build_input_array("cabinet", values)
                    scaled_features = scaler.transform(features)
                    probabilities = model.predict(scaled_features, verbose=0)[0]
                    predicted_label = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))

                    script = generate_cabinet_script(
                        cabinet_height=values["cabinet_height"],
                        cabinet_width=values["cabinet_width"],
                        cabinet_depth=values["cabinet_depth"],
                        wall_thickness=values["wall_thickness"],
                        door_type=values["door_type"],
                        door_count=values["door_count"],
                        style_variant=values["cabinet_style_variant"],
                    )

                    preview_payload = {
                        "object_type": "cabinet",
                        "cabinet_height": values["cabinet_height"],
                        "cabinet_width": values["cabinet_width"],
                        "cabinet_depth": values["cabinet_depth"],
                        "wall_thickness": values["wall_thickness"],
                        "door_type": values["door_type"],
                        "door_count": values["door_count"],
                        "style_variant": values["cabinet_style_variant"],
                        "rotate_yaw": values["rotate_yaw"],
                        "rotate_pitch": values["rotate_pitch"],
                    }
                    preview_image, preview_error = request_preview(preview_payload)
                    preview_src = None
                    if preview_image:
                        preview_src = f"data:image/png;base64,{preview_image}"
                    label_map = LABEL_MAPS["cabinet"]
                else:
                    values["seat_height"] = parse_float("seat_height", values["seat_height"])
                    values["seat_width"] = parse_float("seat_width", values["seat_width"])
                    values["seat_depth"] = parse_float("seat_depth", values["seat_depth"])
                    values["leg_count"] = parse_int("leg_count", values["leg_count"])
                    values["leg_shape"] = parse_str("leg_shape", values["leg_shape"])
                    values["leg_size"] = parse_float("leg_size", values["leg_size"])
                    values["has_backrest"] = parse_int("has_backrest", values["has_backrest"])
                    values["backrest_height"] = parse_float("backrest_height", values["backrest_height"])
                    values["style_variant"] = parse_int("style_variant", values["style_variant"])
                    values["rotate_yaw"] = parse_float("rotate_yaw", values["rotate_yaw"])
                    values["rotate_pitch"] = parse_float("rotate_pitch", values["rotate_pitch"])

                    if values["has_backrest"] == 0:
                        values["backrest_height"] = 0.0
                    else:
                        values["backrest_height"] = max(0.2, min(0.5, values["backrest_height"]))

                    model, scaler = load_artifacts("chair")
                    features = build_input_array("chair", values)
                    scaled_features = scaler.transform(features)
                    probabilities = model.predict(scaled_features, verbose=0)[0]
                    predicted_label = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))

                    script = generate_chair_script(
                        seat_height=values["seat_height"],
                        seat_width=values["seat_width"],
                        seat_depth=values["seat_depth"],
                        leg_count=values["leg_count"],
                        leg_shape=values["leg_shape"],
                        leg_size=values["leg_size"],
                        has_backrest=values["has_backrest"],
                        backrest_height=values["backrest_height"],
                        style_variant=values["style_variant"],
                    )

                    preview_payload = {
                        "object_type": "chair",
                        "seat_height": values["seat_height"],
                        "seat_width": values["seat_width"],
                        "seat_depth": values["seat_depth"],
                        "leg_count": values["leg_count"],
                        "leg_shape": values["leg_shape"],
                        "leg_size": values["leg_size"],
                        "has_backrest": values["has_backrest"],
                        "backrest_height": values["backrest_height"],
                        "style_variant": values["style_variant"],
                        "rotate_yaw": values["rotate_yaw"],
                        "rotate_pitch": values["rotate_pitch"],
                    }
                    preview_image, preview_error = request_preview(preview_payload)
                    preview_src = None
                    if preview_image:
                        preview_src = f"data:image/png;base64,{preview_image}"

                    label_map = LABEL_MAPS["chair"]

                result = {
                        "label": label_map[predicted_label],
                        "confidence": confidence,
                        "probabilities": {label_map[idx]: float(prob) for idx, prob in enumerate(probabilities)},
                    "script": script,
                    "preview_src": preview_src,
                    "preview_error": preview_error,
                }

        return render_template_string(
            HTML_TEMPLATE,
            values=values,
            result=result,
            blender_api_url=BLENDER_API_URL,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
