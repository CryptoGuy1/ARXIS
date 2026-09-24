"""
ARXIS — Central Multimodal Agent Dashboard
Industrial gas safety intelligence: Dueling DQN + LSTM-AE + YOLOv8 + Ollama
"""
import os
import sys
import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# ══════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════
SEED = int(os.environ.get("ARXIS_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
_DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_RAW_CSV = os.environ.get("ARXIS_RAW_CSV", os.path.join(_DEFAULT_DATA_ROOT, "Gas_Sensors_Measurements.csv"))
DEFAULT_IMAGE_DIR = os.environ.get("ARXIS_IMAGE_DIR", os.path.join(_DEFAULT_DATA_ROOT, "Thermal Camera Images"))
DATA_DIR = Path(_DEFAULT_DATA_ROOT)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ARXIS — Industrial Gas Safety Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# LOCAL IMPORTS
# ══════════════════════════════════════════════════════════════
try:
    from src.tools.anomaly_tool import AnomalyTool
    from src.tools.decision_tool import DecisionTool
    from src.tools.explanation_tool import ExplanationTool
    from src.tools.vision_tool import VisionTool
    from src.agent.agent_core import MultimodalAgent
    from src.agent.memory import ShortTermMemory
    from src.agent.goal_manager import GoalManager
    IMPORTS_OK = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
ACTIONS = {0: "Monitor", 1: "Increase Sampling", 2: "Request Verification", 3: "Raise Alarm", 4: "Emergency Shutdown"}
GAS_MAP = {"NoGas": 0, "Smoke": 1, "Mixture": 2, "Perfume": 3}
GAS_NAMES = {v: k for k, v in GAS_MAP.items()}
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
RAW_SENSOR_COLS = ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135"]
IMAGE_NAME_COL_CANDIDATES = ["Corresponding Image Name", "corresponding_image_name", "image_name", "Image Name"]
LABEL_COL_CANDIDATES = ["Gas", "label", "Label", "class", "Class"]

ACTION_COLORS = {0: "#00ff9d", 1: "#3b82f6", 2: "#f59e0b", 3: "#f97316", 4: "#ff2d55"}
GAS_COLORS = {"NoGas": "#00ff9d", "Smoke": "#f97316", "Mixture": "#ff2d55", "Perfume": "#a78bfa"}
ACTION_ICONS = {0: "👁", 1: "📡", 2: "🔍", 3: "🚨", 4: "🔴"}
GAS_ICONS = {"NoGas": "✅", "Smoke": "💨", "Mixture": "☠️", "Perfume": "🌸"}

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

    :root {
        --bg0: #0a0e17;
        --bg1: #0f1520;
        --bg2: #141c2b;
        --bg3: #1a2537;
        --border: #1e3044;
        --border2: #264059;
        --text: #e2e8f0;
        --text2: #94a3b8;
        --muted: #4b5e78;
        --accent: #00d4ff;
        --accent2: #0099cc;
        --amber: #f59e0b;
        --green: #00ff9d;
        --red: #ff2d55;
        --orange: #f97316;
        --blue: #3b82f6;
        --violet: #a78bfa;
        --cyan: #22d3ee;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg0) !important;
        color: var(--text) !important;
    }
    .stApp { background: var(--bg0); }
    .block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1800px; }

    [data-testid="stSidebar"] {
        background: var(--bg1) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stTextInput input {
        background: var(--bg2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        border-radius: 6px !important;
    }

    h1,h2,h3,h4 { font-family: 'Inter', sans-serif !important; font-weight: 700; letter-spacing: -0.02em; }

    /* Header */
    .master-header {
        background: linear-gradient(135deg, var(--bg1) 0%, #0d1825 50%, #081220 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .master-header::after {
        content: '';
        position: absolute;
        top: -100px; right: -100px;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hud-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .hud-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
        letter-spacing: 0.25em;
        margin-top: 8px;
        text-transform: uppercase;
    }
    .hud-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.15em;
        padding: 4px 12px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 8px;
        margin-top: 6px;
        font-weight: 500;
    }
    .badge-online { background: rgba(0,255,157,0.1); border: 1px solid rgba(0,255,157,0.3); color: var(--green); }
    .badge-offline { background: rgba(255,45,85,0.1); border: 1px solid rgba(255,45,85,0.3); color: var(--red); }
    .badge-warn { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); color: var(--amber); }
    .badge-info { background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: var(--accent); }

    /* Section headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--muted);
        letter-spacing: 0.25em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
        padding-bottom: 10px;
        margin: 1.5rem 0 1rem;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, var(--bg2) 0%, var(--bg1) 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--green));
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.2em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .kpi-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .kpi-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: var(--muted);
        margin-top: 8px;
    }

    /* Case card */
    .case-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .case-card-correct { border-left: 4px solid var(--green); }
    .case-card-incorrect { border-left: 4px solid var(--red); }
    .case-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* Action chain chips */
    .action-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 5px 10px;
        border-radius: 5px;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .chip-raw { background: rgba(100,116,139,0.15); border: 1px solid #2d3748; color: var(--text2); }
    .chip-safety { background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25); color: var(--blue); }
    .chip-final-ok { background: rgba(0,255,157,0.1);  border: 1px solid rgba(0,255,157,0.3); color: var(--green); }
    .chip-final-er { background: rgba(255,45,85,0.1);  border: 1px solid rgba(255,45,85,0.3); color: var(--red); }

    /* Info block */
    .info-block {
        background: var(--bg1);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: var(--text2);
        line-height: 2;
        white-space: pre-wrap;
    }

    /* Explanation / Critique wrappers */
    .expl-wrapper {
        background: linear-gradient(135deg, #04101e 0%, #061830 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--cyan);
        border-radius: 0 8px 8px 0;
        padding: 1.4rem;
        margin-top: 0.8rem;
    }
    .crit-wrapper {
        background: linear-gradient(135deg, #1a1500 0%, #221a00 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--amber);
        border-radius: 0 8px 8px 0;
        padding: 1.4rem;
        margin-top: 0.8rem;
    }
    .vision-wrapper {
        background: linear-gradient(135deg, #04140c 0%, #061e14 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--green);
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.4rem;
        margin-top: 0.8rem;
    }

    /* Status strip */
    .status-strip {
        background: var(--bg1);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.8rem;
        display: flex;
        gap: 2rem;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .status-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        color: var(--text2);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .dot-on { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
    .dot-off { background: var(--red); }
    .dot-warn { background: var(--amber); }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg1) !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500;
        letter-spacing: 0.02em !important;
        padding: 0.8rem 1.5rem !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #003d4d 0%, #002933 100%) !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600;
        letter-spacing: 0.05em !important;
        border-radius: 8px !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(0,212,255,0.25) !important;
    }
    .stButton > button:disabled {
        background: var(--bg2) !important;
        border-color: var(--border) !important;
        color: var(--muted) !important;
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--green)) !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        color: var(--text) !important;
    }

    /* Misc */
    .divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def settings_hash(*args):
    key = "|".join(str(a) for a in args)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def infer_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def row_to_sensor_array(row):
    return [float(row[c]) for c in RAW_SENSOR_COLS]


def get_true_gas_id(label):
    if label not in GAS_MAP:
        raise ValueError(f"Unknown label '{label}'")
    return GAS_MAP[label]


def load_raw_dataframe(path):
    if not Path(path).exists():
        st.error(f"Dataset not found: {path}")
        if st.button("🔄 Fetch from Google Drive"):
            try:
                from retrain.fetch_from_drive import main as fetch_main
                fetch_main()
                st.rerun()
            except Exception as e:
                st.error(f"Fetch failed: {e}")
        st.stop()

    df = pd.read_csv(path)
    label_col = infer_column(df, LABEL_COL_CANDIDATES)
    image_col = infer_column(df, IMAGE_NAME_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"No label column found")
    if image_col is None:
        raise ValueError(f"No image-name column found")
    return df, label_col, image_col


def find_matching_target_row(df, label_col, image_col, label, image_path, window_size):
    matches = df[(df[label_col].astype(str) == str(label)) & (df[image_col].astype(str) == str(image_path.stem))]
    if matches.empty:
        raise ValueError(f"No CSV row for label='{label}', image='{image_path.stem}'")
    valid = [i for i in matches.index.tolist() if i >= window_size - 1]
    if not valid:
        raise ValueError(f"No rows with enough history for label='{label}'")
    return valid[0], image_path.stem


def build_window_rows(df, target_idx, window_size=20):
    start_idx = target_idx - window_size + 1
    if start_idx < 0:
        raise ValueError(f"Not enough rows for window_size={window_size}")
    return df.iloc[start_idx:target_idx + 1].copy()


def pick_one_image_per_folder(df, label_col, image_col, image_base_path, window_size):
    """Return dict[label] -> Path, or None if images not available."""
    image_base = Path(image_base_path)
    if not image_base.exists():
        return None
    selected = {}
    for label in ["NoGas", "Smoke", "Mixture", "Perfume"]:
        label_dir = image_base / label
        if not label_dir.exists():
            return None
        images = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if not images:
            return None
        chosen = None
        for image_path in images:
            matches = df[(df[label_col].astype(str) == str(label)) & (df[image_col].astype(str) == str(image_path.stem))]
            if not matches.empty:
                valid = [i for i in matches.index.tolist() if i >= window_size - 1]
                if valid:
                    chosen = image_path
                    break
        if chosen is None:
            return None
        selected[label] = chosen
    return selected


def safe_str(val, fallback="—"):
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s else fallback


def safe_num(val, fmt=".4f", fallback="—"):
    if val is None:
        return fallback
    try:
        if isinstance(val, (list, tuple)):
            val = float(np.mean(val)) if len(val) > 0 else None
            if val is None:
                return fallback
        elif isinstance(val, np.ndarray):
            val = float(np.mean(val))
        else:
            val = float(val)
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def normalize_render_text(text):
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def render_explanation(text, enabled):
    if not enabled:
        content = '<span style="color:#4b5e78;font-style italic;">⚙ Explanations disabled</span>'
    elif text is None or str(text).strip() in ("", "disabled", "None"):
        content = '<span style="color:#4b5e78;font-style italic;">⚠ No explanation returned</span>'
    else:
        cleaned = normalize_render_text(text)
        escaped = cleaned.replace("<", "&lt;").replace(">", "&gt;")
        content = f'<span style="color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:12px;white-space:pre-wrap;">{escaped}</span>'
    st.markdown(f'<div class="expl-wrapper">{content}</div>', unsafe_allow_html=True)


def render_critique(text, enabled):
    if not enabled:
        content = '<span style="color:#4b5e78;font-style italic;">⚙ Critique disabled</span>'
    elif text is None or str(text).strip() in ("", "disabled", "None"):
        content = '<span style="color:#4b5e78;font-style italic;">⚠ No critique returned</span>'
    else:
        cleaned = normalize_render_text(text)
        escaped = cleaned.replace("<", "&lt;").replace(">", "&gt;")
        content = f'<span style="color:#c4a86a;font-family:JetBrains Mono,monospace;font-size:12px;white-space:pre-wrap;">{escaped}</span>'
    st.markdown(f'<div class="crit-wrapper">{content}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# AGENT LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_agent(dqn_path, ae_path, yolo_path, window_size, enable_explanations, enable_critique):
    decision = DecisionTool(dqn_path, device="cpu", mc_dropout_samples=5, window_size=window_size)
    anomaly = AnomalyTool(ae_path)
    vision = VisionTool(yolo_path)
    explainer = None
    if enable_explanations or enable_critique:
        explainer = ExplanationTool("gemma3:1b")
    memory = ShortTermMemory(max_size=200)
    goal_manager = GoalManager()
    agent = MultimodalAgent(
        anomaly_tool=anomaly, decision_tool=decision, explanation_tool=explainer,
        memory=memory, goal_manager=goal_manager, critic=None,
        window_size=window_size, vision_tool=vision,
    )
    return agent


# ══════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,21,32,0.9)",
    font=dict(family="JetBrains Mono, monospace", color="#4b5e78", size=11),
    margin=dict(l=50, r=25, t=50, b=45),
    xaxis=dict(gridcolor="#1e3044", linecolor="#1e3044", zeroline=False),
    yaxis=dict(gridcolor="#1e3044", linecolor="#1e3044", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3044", borderwidth=1),
    title_font=dict(family="Inter, sans-serif", size=14, color="#e2e8f0"),
)


def q_bar_chart(q_values, final_action, label):
    colors_full = [ACTION_COLORS.get(i, "#4b5e78") for i in range(5)]
    bar_colors = [c if i == final_action else f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.2)" for i, c in enumerate(colors_full)]
    fig = go.Figure(go.Bar(
        x=[f"A{i}: {ACTIONS[i]}" for i in range(5)],
        y=q_values,
        marker_color=bar_colors,
        marker_line_color=colors_full,
        marker_line_width=1.5,
        text=[f"{v:.2f}" for v in q_values],
        textposition="outside",
        textfont=dict(size=10, color="#4b5e78"),
    ))
    fig.update_layout(**PLOT_BASE, title=f"Q-Values — {label}", height=280, yaxis_title="Q")
    return fig


def action_dist_chart(logs):
    counts = pd.Series([r["action"] for r in logs]).value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[f"{i}: {ACTIONS[i]}" for i in counts.index],
        y=counts.values,
        marker_color=[ACTION_COLORS.get(i, "#4b5e78") for i in counts.index],
        marker_line_color="#1e3044",
        marker_line_width=1,
        text=counts.values,
        textposition="outside",
        textfont=dict(color="#4b5e78", size=11),
    ))
    fig.update_layout(**PLOT_BASE, title="Final Action Distribution", height=280)
    return fig


def anomaly_scatter(logs):
    df_p = pd.DataFrame(logs)
    fig = go.Figure()
    for lbl, color in GAS_COLORS.items():
        sub = df_p[df_p["label"] == lbl]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["anomaly_normalized"], y=sub["action"],
            mode="markers", name=lbl,
            marker=dict(color=color, size=12, opacity=0.85, line=dict(color=color, width=1.5)),
        ))
    fig.update_layout(**PLOT_BASE, title="Anomaly Score vs Final Action", height=300)
    fig.update_xaxes(title_text="Normalized Anomaly")
    fig.update_yaxes(tickmode="array", tickvals=list(range(5)), ticktext=[f"{i}: {ACTIONS[i]}" for i in range(5)])
    return fig


def anomaly_gauge(anomaly_norm, label):
    val = float(anomaly_norm) if anomaly_norm is not None else 0.0
    bar_color = "#00ff9d" if val < 0.3 else ("#f59e0b" if val < 0.7 else "#ff2d55")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number=dict(font=dict(size=32, family="Inter", color=bar_color)),
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor="#2d3748", tickfont=dict(size=9)),
            bar=dict(color=bar_color),
            bgcolor="rgba(15,21,32,0.9)",
            bordercolor="#1e3044",
            borderwidth=1,
            steps=[
                dict(range=[0, 0.3], color="rgba(0,255,157,0.08)"),
                dict(range=[0.3, 0.7], color="rgba(245,158,11,0.08)"),
                dict(range=[0.7, 1.0], color="rgba(255,45,85,0.1)"),
            ],
            threshold=dict(line=dict(color="#ff2d55", width=2), thickness=0.75, value=0.7),
        ),
    ))
    fig.update_layout(**PLOT_BASE, title=f"Anomaly Gauge — {label}", height=280, margin=dict(l=40, r=40, t=60, b=30))
    return fig


def sensor_traces_chart(step_sensors, label):
    colors_s = ["#ff2d55", "#f97316", "#f59e0b", "#00ff9d", "#3b82f6", "#22d3ee", "#a78bfa"]
    fig = go.Figure()
    arr = np.array(step_sensors)
    for j in range(arr.shape[1]):
        fig.add_trace(go.Scatter(
            x=list(range(arr.shape[0])), y=arr[:, j],
            mode="lines+markers", name=RAW_SENSOR_COLS[j],
            line=dict(color=colors_s[j], width=2), marker=dict(size=3),
        ))
    fig.update_layout(**PLOT_BASE, title=f"Sensor Traces — {label}", height=320)
    fig.update_xaxes(title_text="Window step")
    fig.update_yaxes(title_text="Sensor reading")
    return fig


def anomaly_evolution_chart(step_anomaly_norm, label):
    fig = go.Figure(go.Scatter(
        x=list(range(len(step_anomaly_norm))),
        y=step_anomaly_norm,
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=5, color=[float(v) if v is not None else 0 for v in step_anomaly_norm],
                    colorscale=[[0, "#00ff9d"], [0.5, "#f59e0b"], [1, "#ff2d55"]], cmin=0, cmax=1),
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="#ff2d55",
                 annotation_text="Danger threshold", annotation_position="top right",
                 annotation_font=dict(size=9, color="#ff2d55"))
    fig.update_layout(**PLOT_BASE, title=f"Anomaly Evolution — {label}", height=260)
    fig.update_xaxes(title_text="Window step")
    fig.update_yaxes(title_text="Normalized anomaly", range=[0, 1])
    return fig


def state_vector_chart(state_vector, label):
    colors_st = ["#ff2d55"] + ["#3b82f6"] * 7 + ["#f59e0b"] * 7 + ["#a78bfa"] * 7
    labels_st = (["Anomaly"] + [f"Cur_{c}" for c in RAW_SENSOR_COLS] + [f"Δ_{c}" for c in RAW_SENSOR_COLS] + [f"Std_{c}" for c in RAW_SENSOR_COLS])
    fig = go.Figure(go.Bar(
        x=labels_st, y=state_vector,
        marker_color=colors_st, marker_line_color="#1e3044", marker_line_width=0.5,
        text=[f"{v:.2f}" for v in state_vector], textposition="outside", textfont=dict(size=7, color="#4b5e78"),
    ))
    fig.update_layout(**PLOT_BASE, title=f"State Vector (22-dim) — {label}", height=300, margin=dict(l=48, r=20, t=48, b=80))
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    return fig


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════
def run_single_folder_case(
    agent, df, label_col, image_col, label, image_path, start_idx,
    window_size, use_mc_dropout, enable_explanations, enable_critique,
):
    if image_path is not None:
        target_idx, image_name = find_matching_target_row(
            df=df, label_col=label_col, image_col=image_col,
            label=label, image_path=image_path, window_size=window_size,
        )
    elif start_idx is not None:
        target_idx = start_idx
        image_name = f"sensor-only-{label}"
    else:
        raise ValueError("Either image_path or start_idx must be provided")

    window_df = build_window_rows(df, target_idx, window_size=window_size)
    true_gas_id = get_true_gas_id(label)

    agent.reset_window()
    final_result = None
    step_sensors, step_anomaly_raw, step_anomaly_norm = [], [], []

    for i, (_, row) in enumerate(window_df.iterrows()):
        sensor_array = row_to_sensor_array(row)
        final_image_path = str(image_path) if (image_path is not None and i == (window_size - 1)) else None

        result = agent.run_once(
            sensor_row=sensor_array, step=i, gas_id=true_gas_id,
            image_path=final_image_path, use_mc_dropout=use_mc_dropout,
            enable_explanations=enable_explanations, enable_critique=enable_critique,
        )
        final_result = result
        step_sensors.append(sensor_array)
        step_anomaly_raw.append(result.get("anomaly_raw"))
        step_anomaly_norm.append(result.get("anomaly_normalized"))

    if final_result is None or not final_result.get("ready", False):
        raise RuntimeError(f"Agent did not produce a ready result for '{label}'")

    return {
        "label": label, "image_name": image_name,
        "image_path": str(image_path) if image_path else None,
        "target_idx": int(target_idx),
        "action_raw": final_result.get("action_raw"),
        "action_raw_name": final_result.get("action_raw_name"),
        "action_after_safety": final_result.get("action_after_safety"),
        "action_after_safety_name": final_result.get("action_after_safety_name"),
        "action": final_result.get("action"),
        "action_name": final_result.get("action_name"),
        "gas_id": final_result.get("gas_id"),
        "is_correct": final_result.get("is_correct"),
        "expected_actions": final_result.get("expected_actions"),
        "anomaly_raw": final_result.get("anomaly_raw"),
        "anomaly_normalized": final_result.get("anomaly_normalized"),
        "policy_confidence": final_result.get("policy_confidence"),
        "reward": final_result.get("reward"),
        "latency_ms": float(final_result.get("latency", 0.0) * 1000.0),
        "yolo_class_id": final_result.get("yolo_class_id"),
        "yolo_class_label": final_result.get("yolo_class_label"),
        "yolo_confidence": final_result.get("yolo_confidence"),
        "yolo_semantic_gas_id": final_result.get("yolo_semantic_gas_id"),
        "yolo_gas_name": final_result.get("yolo_gas_name"),
        "vision_action_support": final_result.get("vision_action_support"),
        "vision_danger_flag": final_result.get("vision_danger_flag"),
        "vision_reason": final_result.get("vision_reason"),
        "vision_error": final_result.get("vision_error"),
        "safety_changed_action": final_result.get("safety_changed_action"),
        "vision_escalated_action": final_result.get("vision_escalated_action"),
        "q_values": final_result.get("q_values"),
        "q_std": final_result.get("q_std"),
        "state_vector": final_result.get("state"),
        "explanation": final_result.get("explanation"),
        "critique": final_result.get("critique"),
        "_ran_with_explanations": enable_explanations,
        "_ran_with_critique": enable_critique,
        "_step_sensors": step_sensors,
        "_step_anomaly_raw": step_anomaly_raw,
        "_step_anomaly_norm": step_anomaly_norm,
    }


# ══════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════
inject_css()

# ── HEADER ──────────────────────────────────────────────────
online = IMPORTS_OK
status_html = (
    '<span class="hud-badge badge-online">● ONLINE</span>' if online
    else '<span class="hud-badge badge-offline">✕ IMPORT ERROR</span>'
)

st.markdown(f"""
<div class="master-header">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
    <div>
      <p class="hud-title">⚡ ARXIS</p>
      <p class="hud-subtitle">INDUSTRIAL GAS SAFETY INTELLIGENCE · MULTIMODAL AGENT DASHBOARD</p>
      <div style="margin-top:12px;">
        {status_html}
        <span class="hud-badge badge-info">DUELING DQN</span>
        <span class="hud-badge" style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);color:#f59e0b;">LSTM-AE</span>
        <span class="hud-badge" style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);color:#a78bfa;">YOLOv8</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:JetBrains Mono,monospace; font-size:11px; color:#4b5e78; line-height:2.2;">
        22-FEATURE RL STATE · 5-ACTION POLICY<br>
        MC DROPOUT UNCERTAINTY · REAL-TIME SAFETY<br>
        SENSOR FUSION + VISUAL VERIFICATION
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not IMPORTS_OK:
    st.error(f"**Import failed:** {IMPORT_ERROR}")
    st.stop()


# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚙ Model Paths</p>', unsafe_allow_html=True)
    dqn_model_path = st.text_input("DQN Model", value="models/retrained/exp1_pareto/exp1_miss8_seed42.pth")
    ae_model_path = st.text_input("AE Model", value="models/lstm_autoencoder_weights.pth")
    yolo_model_path = st.text_input("YOLO Model", value="models/yolov8_gas_classifier.pt")
    raw_csv_path = st.text_input("Raw CSV", value=DEFAULT_RAW_CSV)
    image_base_path = st.text_input("Image Folder", value=DEFAULT_IMAGE_DIR)

    st.markdown('<p class="section-header">▶ Run Settings</p>', unsafe_allow_html=True)
    window_size = st.slider("Window Size", 5, 50, 20, 1)
    use_mc = st.checkbox("MC Dropout", value=True)
    show_expl = st.checkbox("Explanations", value=True, help="Requires Ollama running gemma3:1b")
    show_crit = st.checkbox("Critique", value=True, help="Requires ExplanationTool critique pipeline")

    st.markdown('<p class="section-header">◈ Filter</p>', unsafe_allow_html=True)
    filter_label = st.selectbox("Filter by label", ["All", "NoGas", "Smoke", "Mixture", "Perfume"])
    show_only_wrong = st.checkbox("Show only wrong", value=False)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:JetBrains Mono,monospace; font-size:10px; color:#4b5e78; line-height:2.2;">
    <span style="color:#4b5e78;">GAS MAP</span><br>
    <span style="color:#00ff9d;">■</span> NoGas → Monitor<br>
    <span style="color:#f97316;">■</span> Smoke → Raise Alarm<br>
    <span style="color:#ff2d55;">■</span> Mixture → Emergency Shutdown<br>
    <span style="color:#a78bfa;">■</span> Perfume → Increase Sampling
    </div>
    """, unsafe_allow_html=True)


# ── PATH VALIDATION ─────────────────────────────────────────
path_checks = {
    "DQN model": Path(dqn_model_path).exists(),
    "AE model": Path(ae_model_path).exists(),
    "YOLO model": Path(yolo_model_path).exists(),
    "Raw CSV": Path(raw_csv_path).exists(),
    "Image folder": Path(image_base_path).exists(),
}
missing_items = [name for name, ok in path_checks.items() if not ok]

# ── STATUS STRIP ────────────────────────────────────────────
tools_html = ""
for name, ok in path_checks.items():
    dot_cls = "dot-on" if ok else "dot-off"
    tools_html += f'<span class="status-item"><span class="dot {dot_cls}"></span>{name}</span>'
mc_dot = "dot-on" if use_mc else "dot-warn"
tools_html += f'<span class="status-item"><span class="dot {mc_dot}"></span>MC Dropout</span>'
st.markdown(f'<div class="status-strip">{tools_html}</div>', unsafe_allow_html=True)

for item in missing_items:
    st.error(f"⚠ {item} not found — check path in sidebar.")

run_clicked = st.button(
    "▶ RUN FOLDER LIVE TEST" if not missing_items else "⚠ FIX FILE PATHS ABOVE",
    disabled=bool(missing_items),
    use_container_width=True,
)

# ── DATASET PREVIEW ─────────────────────────────────────────
if Path(raw_csv_path).exists():
    try:
        df_preview, lc_preview, _ = load_raw_dataframe(raw_csv_path)
        counts = df_preview[lc_preview].value_counts()
        cols_p = st.columns(min(4, len(counts)))
        for col_, (lbl, cnt) in zip(cols_p, counts.items()):
            color = GAS_COLORS.get(str(lbl), "#4b5e78")
            with col_:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{GAS_ICONS.get(str(lbl), '●')} {lbl}</div>
                  <div class="kpi-value" style="color:{color};">{cnt}</div>
                  <div class="kpi-sub">raw rows</div>
                </div>""", unsafe_allow_html=True)
    except Exception:
        pass

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RUN LOGIC
# ══════════════════════════════════════════════════════════════
current_hash = settings_hash(
    dqn_model_path, ae_model_path, yolo_model_path,
    raw_csv_path, image_base_path, window_size, use_mc, show_expl, show_crit,
)

results_exist = "dashboard_results" in st.session_state
stored_hash = st.session_state.get("settings_hash", None)
settings_changed = results_exist and (stored_hash != current_hash)

if settings_changed:
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:1rem;font-family:JetBrains Mono,monospace;font-size:12px;color:#fbbf24;margin-bottom:1rem;">
    ⚠ Settings changed since last run. Click <strong>RUN FOLDER LIVE TEST</strong> to update.
    </div>
    """, unsafe_allow_html=True)

if run_clicked or results_exist:
    if run_clicked:
        st.session_state.pop("dashboard_results", None)

        with st.spinner("Loading multimodal agent..."):
            agent = load_agent(
                dqn_model_path, ae_model_path, yolo_model_path,
                window_size, show_expl, show_crit,
            )

        with st.spinner("Loading dataset..."):
            df, label_col, image_col = load_raw_dataframe(raw_csv_path)
            chosen_images = pick_one_image_per_folder(
                df=df, label_col=label_col, image_col=image_col,
                image_base_path=image_base_path, window_size=window_size,
            )

        all_results = []
        progress = st.progress(0, text="Running...")

        if chosen_images is None:
            st.info("📷 Images not found — running in **sensor-only mode**")
            chosen_indices = {}
            for label in ["NoGas", "Smoke", "Mixture", "Perfume"]:
                label_df = df[df[label_col] == label]
                if not label_df.empty:
                    chosen_indices[label] = label_df.index[len(label_df) // 2]

            items = list(chosen_indices.items())
            for i, (label, start_idx) in enumerate(items):
                progress.progress((i + 1) / len(items), text=f"Processing {label}...")
                try:
                    result = run_single_folder_case(
                        agent=agent, df=df, label_col=label_col, image_col=image_col,
                        label=label, image_path=None, start_idx=start_idx, window_size=window_size,
                        use_mc_dropout=use_mc, enable_explanations=show_expl, enable_critique=show_crit,
                    )
                    all_results.append(result)
                except Exception as e:
                    st.error(f"Error processing {label}: {e}")
        else:
            items = list(chosen_images.items())
            for i, (label, image_path) in enumerate(items):
                progress.progress((i + 1) / len(items), text=f"Processing {label}...")
                try:
                    result = run_single_folder_case(
                        agent=agent, df=df, label_col=label_col, image_col=image_col,
                        label=label, image_path=image_path, start_idx=None, window_size=window_size,
                        use_mc_dropout=use_mc, enable_explanations=show_expl, enable_critique=show_crit,
                    )
                    all_results.append(result)
                except Exception as e:
                    st.error(f"Error processing {label}: {e}")

        progress.empty()
        st.session_state["dashboard_results"] = all_results
        st.session_state["settings_hash"] = current_hash

    # ── FILTERS ──────────────────────────────────────────────
    logs = st.session_state["dashboard_results"]
    if filter_label != "All":
        logs = [r for r in logs if r["label"] == filter_label]
    if show_only_wrong:
        logs = [r for r in logs if not r["is_correct"]]

    if not logs:
        st.warning("No results match filters.")
        st.stop()

    st.markdown("""
    <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);border-radius:8px;padding:1rem;font-family:JetBrains Mono,monospace;font-size:12px;color:#22d3ee;margin-bottom:1.5rem;">
    ℹ Scope: one case per gas class (n=4). Integration trace, not evaluation. Quantitative results live in <code>retrain/</code>.
    </div>
    """, unsafe_allow_html=True)

    df_logs = pd.DataFrame(logs)

    # ── KPI STRIP ────────────────────────────────────────────
    total = len(df_logs)
    correct = int(df_logs["is_correct"].sum())
    accuracy = correct / total if total else 0.0
    avg_conf = float(df_logs["policy_confidence"].mean()) if total else 0.0
    avg_lat = float(df_logs["latency_ms"].mean()) if total else 0.0

    kc1, kc2, kc3, kc4 = st.columns(4)
    kpi_data = [
        (kc1, "Cases", str(total), "", "#00d4ff"),
        (kc2, "Correct", f"{correct}/{total}", f"{accuracy*100:.1f}%", "#00ff9d" if accuracy == 1 else "#f97316"),
        (kc3, "Avg Confidence", f"{avg_conf:.4f}", "", "#3b82f6"),
        (kc4, "Avg Latency", f"{avg_lat:.1f}", "ms", "#a78bfa"),
    ]
    for col_, label_, val_, sub_, color_ in kpi_data:
        with col_:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label_}</div>
              <div class="kpi-value" style="color:{color_};">{val_}</div>
              <div class="kpi-sub">{sub_}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔬 Case Details", "📈 Sensor Feed", "🧠 State Inspector", "📋 Table"
    ])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(action_dist_chart(logs), use_container_width=True)
        with c2:
            st.plotly_chart(anomaly_scatter(logs), use_container_width=True)

        st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
        for r in logs:
            gc = GAS_COLORS.get(r["label"], "#94a3b8")
            correct_class = "case-card-correct" if r["is_correct"] else "case-card-incorrect"
            final_chip = "chip-final-ok" if r["is_correct"] else "chip-final-er"
            icon = GAS_ICONS.get(r["label"], "●")

            st.markdown(f"""
            <div class="case-card {correct_class}">
              <div class="case-title" style="color:{gc};">{icon} {r['label']} — {r['image_name']}</div>
              <div style="display:flex;align-items:center;gap:8px;margin:12px 0;flex-wrap:wrap;">
                <span class="action-chip chip-raw">DQN: {safe_str(r.get('action_raw_name'))}</span>
                <span style="color:#4b5e78;">→</span>
                <span class="action-chip chip-safety">Safety: {safe_str(r.get('action_after_safety_name'))}</span>
                <span style="color:#4b5e78;">→</span>
                <span class="action-chip {final_chip}">Final: {safe_str(r.get('action_name'))}</span>
              </div>
              <div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#4b5e78;line-height:2;">
                Anomaly: {safe_num(r.get('anomaly_normalized'), '.4f')} &nbsp;|&nbsp;
                Conf: {safe_num(r.get('policy_confidence'), '.4f')} &nbsp;|&nbsp;
                YOLO: {safe_str(r.get('yolo_gas_name'))} ({safe_num(r.get('yolo_confidence'), '.3f')})
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: CASE DETAILS ──────────────────────────────────
    with tab2:
        for r in logs:
            gc = GAS_COLORS.get(r["label"], "#94a3b8")
            icon = GAS_ICONS.get(r["label"], "●")
            correct = "✅ CORRECT" if r["is_correct"] else "❌ WRONG"
            ran_expl = r.get("_ran_with_explanations", show_expl)
            ran_crit = r.get("_ran_with_critique", show_crit)

            with st.expander(f"{icon} {r['label']} · {r['image_name']} · Final: {safe_str(r.get('action_name'))} · {correct}", expanded=True):
                pipeline_col, qval_col = st.columns([1, 1.4])
                with pipeline_col:
                    st.markdown('<p class="section-header">Action Pipeline</p>', unsafe_allow_html=True)
                    stages = ["Raw (DQN)", "After Safety", "Final"]
                    values = [r.get("action_raw"), r.get("action_after_safety"), r.get("action")]
                    labels = [safe_str(r.get("action_raw_name")), safe_str(r.get("action_after_safety_name")), safe_str(r.get("action_name"))]
                    colors = [ACTION_COLORS.get(v, "#4b5e78") for v in values]
                    fig = go.Figure()
                    for i, (stage, label_, color) in enumerate(zip(stages, labels, colors)):
                        fig.add_trace(go.Scatter(
                            x=[i], y=[0], mode="markers+text",
                            marker=dict(size=30, color=color, opacity=0.85, line=dict(color=color, width=2)),
                            text=[f"{ACTION_ICONS.get(values[i], '')}\n{label_}"],
                            textposition="top center", textfont=dict(size=10, color=color), showlegend=False,
                        ))
                        if i < 2:
                            fig.add_annotation(x=i + 0.5, y=0, text="→", showarrow=False, font=dict(size=20, color="#334155"))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=140,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis=dict(visible=False, range=[-0.5, 2.5]),
                        yaxis=dict(visible=False, range=[-0.8, 0.8]),
                        font=dict(family="JetBrains Mono, monospace", color="#94a3b8"),
                    )
                    for i, stage in enumerate(stages):
                        fig.add_annotation(x=i, y=-0.5, text=stage, showarrow=False, font=dict(size=9, color="#4b5e78"), font_family="JetBrains Mono, monospace")
                    st.plotly_chart(fig, use_container_width=True)

                with qval_col:
                    if r.get("q_values") is not None:
                        st.plotly_chart(q_bar_chart(r["q_values"], r["action"], r["label"]), use_container_width=True)
                    else:
                        st.info("Q-values not available.")

                # Structured output
                info_col, vision_col = st.columns(2)
                with info_col:
                    st.markdown('<p class="section-header">Structured Output</p>', unsafe_allow_html=True)
                    anom_n = r.get("anomaly_normalized")
                    anom_r = r.get("anomaly_raw")
                    conf = r.get("policy_confidence")
                    rew = r.get("reward")
                    lat = r.get("latency_ms")
                    q_std = r.get("q_std")
                    q_std_label = f"{safe_num(q_std, '.4f')} (mean)" if isinstance(q_std, (list, tuple, np.ndarray)) else safe_num(q_std, ".4f")
                    exp_act = r.get("expected_actions")
                    exp_act_str = ", ".join(f"{a}={ACTIONS.get(a, a)}" for a in exp_act) if isinstance(exp_act, (list, tuple)) else safe_str(exp_act)

                    st.markdown(f"""<div class="info-block">
Label             : {r['label']}
Image             : {r['image_name']}
CSV target index  : {r['target_idx']}

── Decision Chain ──────────────────────
Raw action        : {safe_str(r.get('action_raw_name'))}
After safety      : {safe_str(r.get('action_after_safety_name'))}
Final action      : {safe_str(r.get('action_name'))}

Correct           : {r['is_correct']}
Expected actions  : {exp_act_str}
Safety changed    : {safe_str(r.get('safety_changed_action'))}
Vision escalated  : {safe_str(r.get('vision_escalated_action'))}

── Sensor / Anomaly ────────────────────
Anomaly (norm)    : {safe_num(anom_n, '.6f')}
Anomaly (raw)     : {safe_num(anom_r, '.6f')}
Policy confidence : {safe_num(conf, '.6f')}
Q-value spread    : {q_std_label}
Reward            : {safe_num(rew, '.4f')}
Latency           : {safe_num(lat, '.2f')} ms

── YOLO Vision ─────────────────────────
Raw class label   : {safe_str(r.get('yolo_class_label'))}
YOLO confidence   : {safe_num(r.get('yolo_confidence'), '.4f')}
Mapped gas name   : {safe_str(r.get('yolo_gas_name'))}
Action support    : {safe_str(r.get('vision_action_support'))}
Danger flag       : {safe_str(r.get('vision_danger_flag'))}</div>""", unsafe_allow_html=True)

                with vision_col:
                    st.markdown('<p class="section-header">Vision Report</p>', unsafe_allow_html=True)
                    if r.get("vision_error"):
                        st.markdown(f'<div class="vision-wrapper" style="border-left-color:#ff2d55;"><span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#ff2d55;">VISION ERROR: {safe_str(r["vision_error"])}</span></div>', unsafe_allow_html=True)
                    else:
                        reason = safe_str(r.get("vision_reason"), "No vision reason returned.")
                        st.markdown(f'<div class="vision-wrapper"><span style="font-family:JetBrains Mono,monospace;font-size:12px;color:#7ab89a;">{reason}</span></div>', unsafe_allow_html=True)

                st.markdown('<p class="section-header">🧠 Explanation</p>', unsafe_allow_html=True)
                render_explanation(r.get("explanation"), ran_expl)

                st.markdown('<p class="section-header">⚖ Critique</p>', unsafe_allow_html=True)
                render_critique(r.get("critique"), ran_crit)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── TAB 3: SENSOR FEED ───────────────────────────────────
    with tab3:
        st.markdown('<p class="section-header">Multi-Channel Sensor Feed</p>', unsafe_allow_html=True)
        gas_labels = [r["label"] for r in logs]
        selected_gas = st.selectbox("Select gas case", gas_labels, key="sensor_gas_sel")
        r = next(x for x in logs if x["label"] == selected_gas)

        if r.get("_step_sensors"):
            st.plotly_chart(sensor_traces_chart(r["_step_sensors"], r["label"]), use_container_width=True)
        if r.get("_step_anomaly_norm"):
            st.plotly_chart(anomaly_evolution_chart(r["_step_anomaly_norm"], r["label"]), use_container_width=True)

        if r.get("_step_sensors"):
            arr = np.array(r["_step_sensors"])
            st.markdown('<p class="section-header">Per-Sensor Summary</p>', unsafe_allow_html=True)
            scols = st.columns(7)
            for j, (col, name) in enumerate(zip(scols, RAW_SENSOR_COLS)):
                with col:
                    mean_v = float(arr[:, j].mean())
                    std_v = float(arr[:, j].std())
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-label">{name}</div>
                      <div class="kpi-value" style="color:#3b82f6;font-size:1.4rem;">{mean_v:.1f}</div>
                      <div class="kpi-sub">± {std_v:.1f}</div>
                    </div>""", unsafe_allow_html=True)

    # ── TAB 4: STATE INSPECTOR ───────────────────────────────
    with tab4:
        st.markdown('<p class="section-header">DQN State Vector Inspector</p>', unsafe_allow_html=True)
        gas_labels_2 = [r["label"] for r in logs]
        selected_gas_2 = st.selectbox("Select gas case", gas_labels_2, key="state_gas_sel")
        r2 = next(x for x in logs if x["label"] == selected_gas_2)

        if r2.get("anomaly_normalized") is not None:
            st.plotly_chart(anomaly_gauge(r2["anomaly_normalized"], r2["label"]), use_container_width=True)

        if r2.get("state_vector") is not None:
            sv = r2["state_vector"]
            if isinstance(sv, np.ndarray) and sv.shape[0] == 22:
                st.plotly_chart(state_vector_chart(sv, r2["label"]), use_container_width=True)
            elif isinstance(sv, (list, tuple)) and len(sv) == 22:
                st.plotly_chart(state_vector_chart(np.array(sv), r2["label"]), use_container_width=True)

        # Decision chain
        st.markdown('<p class="section-header">Decision Chain</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="case-card">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span class="action-chip chip-raw">DQN: {safe_str(r2.get('action_raw_name'))}</span>
            <span style="color:#4b5e78;">→</span>
            <span class="action-chip chip-safety">Safety: {safe_str(r2.get('action_after_safety_name'))}</span>
            <span style="color:#4b5e78;">→</span>
            <span class="action-chip chip-final-ok">Final: {safe_str(r2.get('action_name'))}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── TAB 5: STRUCTURED TABLE ──────────────────────────────
    with tab5:
        st.markdown('<p class="section-header">Export-Ready Results</p>', unsafe_allow_html=True)
        keep_cols = [
            "label", "image_name", "target_idx",
            "action_raw_name", "action_after_safety_name", "action_name",
            "is_correct", "policy_confidence", "anomaly_normalized", "reward",
            "yolo_class_label", "yolo_confidence", "yolo_gas_name",
            "vision_action_support", "vision_danger_flag",
            "safety_changed_action", "vision_escalated_action", "latency_ms",
        ]
        table_df = df_logs[[c for c in keep_cols if c in df_logs.columns]].copy()
        st.dataframe(table_df, use_container_width=True, height=400)
