import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile as tiff
import os
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep UNet · Satellite Segmentation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES = [
    {"name": "Buildings",    "hex": "#969696", "rgb": (150, 150, 150), "weight": 0.20, "icon": "🏗️"},
    {"name": "Roads",        "hex": "#DFC27D", "rgb": (223, 194, 125), "weight": 0.30, "icon": "🛣️"},
    {"name": "Trees",        "hex": "#1B7837", "rgb": (27,  120,  55), "weight": 0.10, "icon": "🌳"},
    {"name": "Crops",        "hex": "#A6DBA0", "rgb": (166, 219, 160), "weight": 0.10, "icon": "🌾"},
    {"name": "Water",        "hex": "#74ADD1", "rgb": (116, 173, 209), "weight": 0.30, "icon": "💧"},
]

BANDS = [
    ("Coastal",  "~450 nm", "#4a90d9"),
    ("Blue",     "~480 nm", "#5b9bd5"),
    ("Green",    "~560 nm", "#4dab4d"),
    ("Yellow",   "~590 nm", "#e0b840"),
    ("Red",      "~625 nm", "#e05050"),
    ("Red Edge", "~700 nm", "#c04090"),
    ("Near-IR1", "~780 nm", "#8b508b"),
    ("Near-IR2", "~870 nm", "#5b3092"),
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_IDS = [str(i).zfill(2) for i in range(1, 25)]
LOG_PATH  = os.path.join(os.path.dirname(__file__), "log_unet.csv")

ACCENT   = "#00B4D8"
ACCENT2  = "#7c3aed"
BG_CARD  = "rgba(255,255,255,0.04)"
BORDER   = "rgba(0,180,216,0.25)"

# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── font & global ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── sidebar ───────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050810 0%, #080C18 100%);
        border-right: 1px solid rgba(0,180,216,0.15);
    }
    section[data-testid="stSidebar"] * { color: #CDD6F4 !important; }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 6px 10px;
        border-radius: 8px;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0,180,216,0.12);
    }

    /* ── main area ─────────────────────────────────────────────────── */
    .main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }
    .stApp { background-color: #080C18; }

    /* ── hero banner ───────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #050810 0%, #0a1628 40%, #071220 100%);
        border: 1px solid rgba(0,180,216,0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,180,216,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 40% 60% at 20% 20%, rgba(124,58,237,0.06) 0%, transparent 50%);
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 700; line-height: 1.2;
        background: linear-gradient(90deg, #00B4D8, #7c3aed, #00B4D8);
        background-size: 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s linear infinite;
        position: relative;
    }
    @keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
    .hero-sub { color: #8892a4; font-size: 1rem; margin-top: 0.4rem; position: relative; }
    .hero-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem; position: relative; }
    .badge {
        background: rgba(0,180,216,0.12); border: 1px solid rgba(0,180,216,0.3);
        color: #00B4D8; border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; font-weight: 500;
    }

    /* ── metric cards ──────────────────────────────────────────────── */
    .metric-row { display: flex; gap: 14px; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 140px;
        background: linear-gradient(135deg, rgba(0,180,216,0.08), rgba(13,22,38,0.9));
        border: 1px solid rgba(0,180,216,0.2);
        border-radius: 14px; padding: 1.2rem 1.4rem;
        transition: border-color 0.2s, transform 0.2s;
    }
    .metric-card:hover { border-color: rgba(0,180,216,0.5); transform: translateY(-2px); }
    .metric-card .val {
        font-size: 2rem; font-weight: 700; color: #00B4D8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .lbl { font-size: 0.8rem; color: #6e7a8a; margin-top: 2px; }
    .metric-card .icon { font-size: 1.6rem; margin-bottom: 0.4rem; }

    /* ── section headers ───────────────────────────────────────────── */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.1rem; font-weight: 600; color: #CDD6F4;
        margin: 1.5rem 0 1rem;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0,180,216,0.2);
    }
    .section-header .dot { width:8px;height:8px;border-radius:50%; background:#00B4D8; }

    /* ── glass cards ───────────────────────────────────────────────── */
    .glass-card {
        background: rgba(13,22,38,0.7);
        border: 1px solid rgba(0,180,216,0.15);
        border-radius: 14px; padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    /* ── class legend ──────────────────────────────────────────────── */
    .legend-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
    .legend-item {
        display: flex; align-items: center; gap: 10px;
        background: rgba(255,255,255,0.04); border-radius: 10px; padding: 10px 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .legend-swatch { width:16px; height:16px; border-radius:4px; flex-shrink:0; }
    .legend-name { font-size: 0.85rem; font-weight: 500; color: #CDD6F4; }
    .legend-weight { font-size: 0.75rem; color: #6e7a8a; }

    /* ── band chips ─────────────────────────────────────────────────── */
    .band-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .band-chip {
        border-radius: 10px; padding: 10px 12px; text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .band-chip .band-name { font-size: 0.82rem; font-weight: 600; }
    .band-chip .band-wl  { font-size: 0.72rem; color: #6e7a8a; margin-top: 2px; }

    /* ── status pill ────────────────────────────────────────────────── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; font-weight: 500;
    }
    .status-ok   { background: rgba(74,222,128,0.12); border:1px solid rgba(74,222,128,0.3); color:#4ade80; }
    .status-warn { background: rgba(251,191, 36,0.12); border:1px solid rgba(251,191,36,0.3);  color:#fbbf24; }
    .status-dot  { width:6px; height:6px; border-radius:50%; background:currentColor; }

    /* ── tabs override ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(0,180,216,0.2); }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 8px 20px;
        background: transparent; color: #6e7a8a;
        border: none; transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,180,216,0.12) !important;
        color: #00B4D8 !important;
        border-bottom: 2px solid #00B4D8;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #CDD6F4; }

    /* ── selectbox / slider tweaks ──────────────────────────────────── */
    .stSelectbox label, .stSlider label { color: #8892a4 !important; font-size: 0.85rem !important; }

    /* ── file uploader ───────────────────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(0,180,216,0.04) !important;
        border: 2px dashed rgba(0,180,216,0.35) !important;
        border-radius: 12px !important;
    }

    /* ── divider ─────────────────────────────────────────────────────── */
    hr { border-color: rgba(0,180,216,0.12) !important; }

    /* ── image caption ───────────────────────────────────────────────── */
    .img-caption {
        text-align: center; font-size: 0.78rem; color: #6e7a8a;
        margin-top: 4px;
    }

    /* ── arch layer box ───────────────────────────────────────────────── */
    .arch-row { display:flex; align-items:center; justify-content:center; gap:8px; margin:4px 0; }
    .arch-box {
        border-radius:8px; padding:6px 16px; font-size:0.78rem; font-weight:500;
        text-align:center; border:1px solid;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_training_log():
    if not os.path.exists(LOG_PATH):
        return None
    df = pd.read_csv(LOG_PATH, sep=";")
    df.columns = df.columns.str.strip()
    df["epoch"] = df["epoch"].astype(int) + 1
    return df


@st.cache_data(show_spinner=False)
def load_image(img_id):
    path = os.path.join(DATA_DIR, "mband", f"{img_id}.tif")
    if not os.path.exists(path):
        return None
    img = tiff.imread(path)
    if img.ndim == 3 and img.shape[0] <= 10:
        img = img.transpose(1, 2, 0)
    return img.astype(np.float32)


@st.cache_data(show_spinner=False)
def load_mask(img_id):
    path = os.path.join(DATA_DIR, "gt_mband", f"{img_id}.tif")
    if not os.path.exists(path):
        return None
    mask = tiff.imread(path)
    if mask.ndim == 3 and mask.shape[0] <= 10:
        mask = mask.transpose(1, 2, 0)
    return (mask / 255.0).astype(np.float32)


def normalize_img(img):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            out[:, :, c] = (ch - mn) / (mx - mn)
        else:
            out[:, :, c] = 0.0
    return out


def false_color_rgb(img, r_band=4, g_band=2, b_band=1):
    norm = normalize_img(img)
    r = norm[:, :, r_band]
    g = norm[:, :, g_band]
    b = norm[:, :, b_band]
    rgb = np.stack([r, g, b], axis=2)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def mask_to_rgb(mask, threshold=0.5):
    h, w = mask.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    z_order = [2, 3, 0, 1, 4]
    for z, cls_idx in enumerate(z_order):
        if cls_idx >= mask.shape[2]:
            continue
        rgb = CLASSES[cls_idx]["rgb"]
        hit = mask[:, :, cls_idx] > threshold
        for ch in range(3):
            canvas[:, :, ch][hit] = rgb[ch]
    return canvas


def fig_to_image(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def section_header(icon, title):
    st.markdown(
        f'<div class="section-header"><span class="dot"></span>{icon} {title}</div>',
        unsafe_allow_html=True,
    )


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_dashboard():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🛰️ Deep UNet · Satellite Segmentation</div>
        <div class="hero-sub">
            Pixel-level semantic segmentation of 8-band multispectral satellite imagery
            using a deep encoder-decoder U-Net trained on the SpaceNet dataset.
        </div>
        <div class="hero-badges">
            <span class="badge">Keras / TensorFlow</span>
            <span class="badge">SpaceNet Dataset</span>
            <span class="badge">8-Band Multispectral</span>
            <span class="badge">5-Class Segmentation</span>
            <span class="badge">Tesla P100 · 16 GB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    df = load_training_log()
    best_val = f"{df['val_loss'].min():.4f}" if df is not None else "N/A"
    final_tr  = f"{df['loss'].iloc[-1]:.4f}"  if df is not None else "N/A"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="icon">🗂️</div>
            <div class="val">24</div>
            <div class="lbl">Training Locations</div>
        </div>
        <div class="metric-card">
            <div class="icon">🏷️</div>
            <div class="val">5</div>
            <div class="lbl">Semantic Classes</div>
        </div>
        <div class="metric-card">
            <div class="icon">📡</div>
            <div class="val">8</div>
            <div class="lbl">Spectral Bands</div>
        </div>
        <div class="metric-card">
            <div class="icon">🔄</div>
            <div class="val">150</div>
            <div class="lbl">Training Epochs</div>
        </div>
        <div class="metric-card">
            <div class="icon">📉</div>
            <div class="val">{best_val}</div>
            <div class="lbl">Best Val Loss</div>
        </div>
        <div class="metric-card">
            <div class="icon">📊</div>
            <div class="val">{final_tr}</div>
            <div class="lbl">Final Train Loss</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_right = st.columns([3, 2], gap="large")

    with col_chart:
        section_header("📈", "Training History")
        if df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df["loss"],
                mode="lines", name="Train Loss",
                line=dict(color="#00B4D8", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,180,216,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df["val_loss"],
                mode="lines", name="Val Loss",
                line=dict(color="#7c3aed", width=2.5, dash="dot"),
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.05)",
            ))
            # Best val line
            best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
            fig.add_vline(
                x=best_epoch, line_color="rgba(74,222,128,0.4)",
                line_dash="dash", line_width=1.5,
                annotation_text=f"best @ ep{best_epoch}",
                annotation_font_color="#4ade80",
                annotation_font_size=11,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8892a4", size=12),
                legend=dict(
                    orientation="h", yanchor="top", y=1.12,
                    bgcolor="rgba(0,0,0,0)", font_color="#CDD6F4",
                ),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    title="Epoch", title_font_color="#6e7a8a",
                    tickcolor="#6e7a8a",
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    title="Weighted BCE Loss", title_font_color="#6e7a8a",
                    tickcolor="#6e7a8a",
                ),
                margin=dict(l=10, r=10, t=30, b=10),
                height=320,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("log_unet.csv not found — train the model to see history.")

    with col_right:
        section_header("🏷️", "Class Weights")
        names    = [c["name"]   for c in CLASSES]
        weights  = [c["weight"] for c in CLASSES]
        colors   = [c["hex"]    for c in CLASSES]
        fig2 = go.Figure(go.Bar(
            x=weights, y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{w:.0%}" for w in weights],
            textposition="outside",
            textfont=dict(color="#CDD6F4", size=12),
        ))
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892a4"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#CDD6F4", size=13)),
            margin=dict(l=10, r=50, t=10, b=10),
            height=250,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        section_header("🌈", "Class Legend")
        legend_html = '<div class="legend-grid">'
        for cls in CLASSES:
            legend_html += f"""
            <div class="legend-item">
                <div class="legend-swatch" style="background:{cls['hex']};"></div>
                <div>
                    <div class="legend-name">{cls['icon']} {cls['name']}</div>
                    <div class="legend-weight">weight {cls['weight']:.2f}</div>
                </div>
            </div>"""
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    # Spectral bands
    section_header("📡", "Spectral Bands")
    bands_html = '<div class="band-grid">'
    for i, (name, wl, color) in enumerate(BANDS):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        bands_html += f"""
        <div class="band-chip" style="background:rgba({r},{g},{b},0.12);border-color:rgba({r},{g},{b},0.35);">
            <div class="band-name" style="color:{color};">Band {i+1} · {name}</div>
            <div class="band-wl">{wl}</div>
        </div>"""
    bands_html += "</div>"
    st.markdown(bands_html, unsafe_allow_html=True)


def page_data_explorer():
    section_header("🗺️", "Data Explorer")
    st.markdown('<p style="color:#8892a4;font-size:0.9rem;margin-bottom:1rem;">Browse the 24 multispectral training scenes and their ground-truth segmentation masks.</p>', unsafe_allow_html=True)

    # Controls
    c1, c2, c3 = st.columns([1.5, 1.5, 3])
    with c1:
        img_id = st.selectbox("Scene ID", TRAIN_IDS, format_func=lambda x: f"Scene {x}")
    with c2:
        view_mode = st.selectbox("View", ["False Color", "Single Band", "Mask"])
    with c3:
        band_idx = 0
        if view_mode == "Single Band":
            band_names = [f"Band {i+1} · {b[0]}" for i, b in enumerate(BANDS)]
            band_sel = st.selectbox("Band", band_names, index=4)
            band_idx = band_names.index(band_sel)

    img  = load_image(img_id)
    mask = load_mask(img_id)

    if img is None:
        st.warning(f"Image not found: data/mband/{img_id}.tif")
        return

    st.markdown("---")
    col_img, col_mask = st.columns(2, gap="large")

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if view_mode == "False Color":
            rgb = false_color_rgb(img, r_band=4, g_band=2, b_band=1)
            fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0D1626")
            ax.imshow(rgb)
            ax.set_title(f"Scene {img_id} · False Color (B5-B3-B2)", color="#CDD6F4", fontsize=10)
            ax.axis("off")
            fig.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('<div class="img-caption">Bands 5,3,2 mapped to R,G,B · 16-bit normalized</div>', unsafe_allow_html=True)

        elif view_mode == "Single Band":
            norm = normalize_img(img)
            ch   = norm[:, :, band_idx]
            bname, bwl, bcolor = BANDS[band_idx]
            fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0D1626")
            im = ax.imshow(ch, cmap="inferno")
            ax.set_title(f"Scene {img_id} · Band {band_idx+1}: {bname} ({bwl})", color="#CDD6F4", fontsize=10)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(colors="#6e7a8a", labelsize=8)
            fig.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(f'<div class="img-caption">{bname} · {bwl}</div>', unsafe_allow_html=True)

        else:
            if mask is not None:
                mask_rgb = mask_to_rgb(mask)
                fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0D1626")
                ax.imshow(mask_rgb)
                ax.set_title(f"Scene {img_id} · Ground-Truth Mask", color="#CDD6F4", fontsize=10)
                ax.axis("off")
                patches = [mpatches.Patch(color=np.array(c["rgb"])/255, label=c["name"]) for c in CLASSES]
                ax.legend(handles=patches, loc="lower right", fontsize=7,
                          framealpha=0.5, facecolor="#0D1626", edgecolor="#333", labelcolor="#CDD6F4")
                fig.tight_layout(pad=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("Mask not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_mask:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if mask is not None:
            # Per-class coverage pie chart
            coverage = []
            total_px = mask.shape[0] * mask.shape[1]
            for i, cls in enumerate(CLASSES):
                frac = float((mask[:, :, i] > 0.5).sum()) / total_px
                coverage.append(frac)
            others = max(0.0, 1.0 - sum(coverage))

            labels = [c["name"] for c in CLASSES] + ["Background"]
            values = coverage + [others]
            clrs   = [c["hex"] for c in CLASSES] + ["#1a2030"]

            fig3 = go.Figure(go.Pie(
                labels=labels, values=values,
                marker=dict(colors=clrs, line=dict(color="#080C18", width=2)),
                textinfo="percent+label",
                textfont=dict(size=11, color="#CDD6F4"),
                hole=0.45,
            ))
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8892a4"),
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
                title=dict(text=f"Scene {img_id} · Class Coverage", font=dict(color="#CDD6F4", size=11)),
                annotations=[dict(text="Coverage", x=0.5, y=0.5, font_size=12,
                                  font_color="#6e7a8a", showarrow=False)],
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

            # Band histogram
            st.markdown('<div class="section-header" style="font-size:0.9rem;margin:0.5rem 0 0.5rem;"><span class="dot"></span>Band Intensity Distribution</div>', unsafe_allow_html=True)
            norm2 = normalize_img(img)
            fig4, ax4 = plt.subplots(figsize=(5, 2.2), facecolor="#0D1626")
            ax4.set_facecolor("#0D1626")
            for i, (bname, _, bcolor) in enumerate(BANDS):
                vals = norm2[:, :, i].ravel()
                ax4.hist(vals, bins=60, alpha=0.55, color=bcolor, histtype="step",
                         linewidth=1.5, label=bname)
            ax4.set_xlabel("Normalized Intensity", color="#6e7a8a", fontsize=8)
            ax4.set_ylabel("Pixels", color="#6e7a8a", fontsize=8)
            ax4.tick_params(colors="#6e7a8a", labelsize=7)
            ax4.spines[:].set_color("rgba(255,255,255,0.08)")
            leg = ax4.legend(fontsize=7, ncol=2, framealpha=0.3,
                             facecolor="#0D1626", edgecolor="#333", labelcolor="#CDD6F4")
            fig4.tight_layout(pad=0.3)
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)
        else:
            st.info("Ground-truth mask not available for this scene.")
        st.markdown('</div>', unsafe_allow_html=True)


def page_inference():
    section_header("⚡", "Run Inference")

    weights_exist = os.path.exists(os.path.join(os.path.dirname(__file__), "weights", "unet_weights.hdf5"))

    if weights_exist:
        st.markdown('<span class="status-pill status-ok"><span class="status-dot"></span>Model weights found</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-warn"><span class="status-dot"></span>Weights not found — demo mode active</span>', unsafe_allow_html=True)

    st.markdown("")

    tabs = st.tabs(["📁 Upload Image", "🔍 Use Training Sample"])

    with tabs[0]:
        st.markdown('<p style="color:#8892a4;font-size:0.88rem;">Upload an 8-band multispectral GeoTIFF file (shape: bands × H × W or H × W × bands).</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop your .tif file here", type=["tif", "tiff"])

        if uploaded:
            raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
            try:
                img = tiff.imread(io.BytesIO(raw))
            except Exception:
                st.error("Could not parse the TIFF file.")
                return

            if img.ndim == 3 and img.shape[0] <= 10:
                img = img.transpose(1, 2, 0)
            img = img.astype(np.float32)

            st.success(f"Loaded: {img.shape[1]}×{img.shape[0]} px · {img.shape[2]} bands")
            _show_inference_panel(img, has_weights=weights_exist)

    with tabs[1]:
        sample_id = st.selectbox("Choose a training scene", TRAIN_IDS, format_func=lambda x: f"Scene {x}", key="inf_sample")
        img = load_image(sample_id)
        if img is None:
            st.warning("Scene image not found in data/mband/.")
        else:
            _show_inference_panel(img, has_weights=weights_exist, mask_id=sample_id)


def _show_inference_panel(img, has_weights=False, mask_id=None):
    col_in, col_out = st.columns(2, gap="large")

    with col_in:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        rgb = false_color_rgb(img, r_band=min(4, img.shape[2]-1),
                              g_band=min(2, img.shape[2]-1),
                              b_band=min(1, img.shape[2]-1))
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0D1626")
        ax.imshow(rgb)
        ax.set_title("Input · False Color (B5-B3-B2)", color="#CDD6F4", fontsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('<div class="img-caption">Normalized 8-band multispectral input</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if has_weights:
            with st.spinner("Running inference with TTA (7 augmentations)…"):
                try:
                    import sys
                    sys.path.insert(0, os.path.dirname(__file__))
                    from train_unet import get_model, normalize, PATCH_SZ, N_CLASSES
                    from predict import predict, picture_from_mask
                    weights_path = os.path.join(os.path.dirname(__file__), "weights", "unet_weights.hdf5")
                    model = get_model()
                    model.load_weights(weights_path)
                    norm_img = normalize(img)
                    pred = predict(norm_img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
                    pred_t = pred.transpose([2, 0, 1])
                    map_rgb = picture_from_mask(pred_t, 0.5).transpose(1, 2, 0)
                    fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor="#0D1626")
                    ax2.imshow(map_rgb)
                    ax2.set_title("Predicted Segmentation Map", color="#CDD6F4", fontsize=10)
                    ax2.axis("off")
                    patches = [mpatches.Patch(color=np.array(c["rgb"])/255, label=c["name"]) for c in CLASSES]
                    ax2.legend(handles=patches, loc="lower right", fontsize=7,
                               framealpha=0.5, facecolor="#0D1626", edgecolor="#333", labelcolor="#CDD6F4")
                    fig2.tight_layout(pad=0.3)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
        else:
            # Demo: show ground-truth or a noise placeholder
            if mask_id:
                mask = load_mask(mask_id)
            else:
                mask = None

            if mask is not None:
                mask_rgb = mask_to_rgb(mask)
                fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor="#0D1626")
                ax2.imshow(mask_rgb)
                ax2.set_title("Ground-Truth Mask (demo — no weights)", color="#fbbf24", fontsize=10)
                ax2.axis("off")
                patches = [mpatches.Patch(color=np.array(c["rgb"])/255, label=c["name"]) for c in CLASSES]
                ax2.legend(handles=patches, loc="lower right", fontsize=7,
                           framealpha=0.5, facecolor="#0D1626", edgecolor="#333", labelcolor="#CDD6F4")
                fig2.tight_layout(pad=0.3)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            else:
                st.info("No pre-trained weights found.\nTrain the model first:\n```\npython train_unet.py\n```")
        st.markdown('</div>', unsafe_allow_html=True)

    # Per-class confidence bars (demo: random, or real from prediction)
    if has_weights and mask_id is None:
        pass
    elif mask_id:
        mask = load_mask(mask_id)
        if mask is not None:
            section_header("📊", "Per-Class Pixel Coverage")
            total_px = mask.shape[0] * mask.shape[1]
            cols = st.columns(5)
            for i, (cls, col) in enumerate(zip(CLASSES, cols)):
                frac = float((mask[:, :, i] > 0.5).sum()) / total_px
                with col:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;padding:0.8rem 0.5rem;">
                        <div style="font-size:1.6rem;">{cls['icon']}</div>
                        <div style="font-size:0.82rem;color:#CDD6F4;font-weight:600;">{cls['name']}</div>
                        <div style="font-size:1.2rem;font-weight:700;color:{cls['hex']};
                                    font-family:'JetBrains Mono',monospace;">{frac:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)


def page_architecture():
    section_header("🏗️", "Model Architecture")

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Draw U-Net schematic with matplotlib
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0D1626")
        ax.set_facecolor("#0D1626")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title("Deep U-Net Architecture (6-level encoder-decoder)", color="#CDD6F4", fontsize=11, pad=12)

        enc_color  = "#00B4D8"
        dec_color  = "#7c3aed"
        skip_color = "#4ade80"
        bot_color  = "#f59e0b"

        # Encoder levels: (x, y, label, n_filters)
        encoder = [
            (1.5, 6.8, "Conv 32×2\nInput 160×160×8",  32),
            (1.5, 5.6, "Conv 64×2 + BN\nPool + Drop",   64),
            (1.5, 4.4, "Conv 128×2 + BN\nPool + Drop",  128),
            (1.5, 3.2, "Conv 256×2 + BN\nPool + Drop",  256),
            (1.5, 2.0, "Conv 512×2 + BN\nPool + Drop",  512),
        ]
        # Bottleneck
        bottleneck = (5.0, 0.9, "Conv 1024×2\nBottleneck", 1024)
        # Decoder
        decoder = [
            (8.5, 2.0, "UpConv 512\n+ Skip + BN + Drop", 512),
            (8.5, 3.2, "UpConv 256\n+ Skip + BN + Drop", 256),
            (8.5, 4.4, "UpConv 128\n+ Skip + BN + Drop", 128),
            (8.5, 5.6, "UpConv 64\n+ Skip + BN + Drop",   64),
            (8.5, 6.8, "Conv 32×2\nOutput 160×160×5",     32),
        ]

        def draw_box(ax, x, y, label, color, width=2.0, height=0.7):
            rect = plt.Rectangle((x - width/2, y - height/2), width, height,
                                  linewidth=1.5, edgecolor=color,
                                  facecolor=f"{color}18", zorder=3)
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center",
                    color=color, fontsize=6.5, zorder=4,
                    fontfamily="monospace", linespacing=1.4)

        for ex, ey, elbl, _ in encoder:
            draw_box(ax, ex, ey, elbl, enc_color)
        draw_box(ax, bottleneck[0], bottleneck[1], bottleneck[2], bot_color, width=2.4, height=0.7)
        for dx, dy, dlbl, _ in decoder:
            draw_box(ax, dx, dy, dlbl, dec_color)

        # Skip connections
        for (ex, ey, _, __), (dx, dy, ___, ____) in zip(encoder, reversed(decoder)):
            ax.annotate("", xy=(dx - 1.0, dy), xytext=(ex + 1.0, ey),
                        arrowprops=dict(arrowstyle="-|>", color=skip_color,
                                        lw=1.2, connectionstyle="arc3,rad=-0.3"),
                        zorder=2)

        # Encoder downward arrows
        for i in range(len(encoder) - 1):
            x1, y1 = encoder[i][0], encoder[i][1]
            x2, y2 = encoder[i+1][0], encoder[i+1][1]
            ax.annotate("", xy=(x2, y2 + 0.35), xytext=(x1, y1 - 0.35),
                        arrowprops=dict(arrowstyle="-|>", color=enc_color, lw=1.2), zorder=2)

        # Encoder → bottleneck
        ax.annotate("", xy=(bottleneck[0] - 1.2, bottleneck[1] + 0.35),
                    xytext=(encoder[-1][0], encoder[-1][1] - 0.35),
                    arrowprops=dict(arrowstyle="-|>", color=bot_color, lw=1.2), zorder=2)

        # Bottleneck → first decoder
        ax.annotate("", xy=(decoder[0][0] - 1.0, decoder[0][1] - 0.35),
                    xytext=(bottleneck[0] + 1.2, bottleneck[1] + 0.35),
                    arrowprops=dict(arrowstyle="-|>", color=dec_color, lw=1.2), zorder=2)

        # Decoder upward arrows
        for i in range(len(decoder) - 1):
            x1, y1 = decoder[i][0], decoder[i][1]
            x2, y2 = decoder[i+1][0], decoder[i+1][1]
            ax.annotate("", xy=(x2, y2 - 0.35), xytext=(x1, y1 + 0.35),
                        arrowprops=dict(arrowstyle="-|>", color=dec_color, lw=1.2), zorder=2)

        # Legend
        legend_items = [
            mpatches.Patch(color=enc_color, label="Encoder"),
            mpatches.Patch(color=bot_color, label="Bottleneck"),
            mpatches.Patch(color=dec_color, label="Decoder"),
            mpatches.Patch(color=skip_color, label="Skip Connection"),
        ]
        ax.legend(handles=legend_items, loc="lower center", ncol=4,
                  fontsize=8, framealpha=0.4, facecolor="#0D1626",
                  edgecolor="#333", labelcolor="#CDD6F4")

        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        section_header("⚙️", "Configuration")

        config_data = {
            "Parameter": [
                "Input Shape", "Classes", "Channels", "Base Filters",
                "Growth Factor", "Depth (encoder)", "Up-Convolution",
                "Dropout Rate", "Optimizer", "Loss Function",
                "Patch Size", "Batch Size",
            ],
            "Value": [
                "160 × 160 × 8", "5", "8", "32",
                "×2", "6 levels", "Transposed Conv",
                "0.25", "Adam (lr=1e-3)", "Weighted BCE",
                "160 px", "150",
            ],
        }
        df_cfg = pd.DataFrame(config_data)
        st.dataframe(
            df_cfg,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                "Value":     st.column_config.TextColumn("Value",     width="small"),
            },
        )

        section_header("📐", "Filter Progression")
        filters = [32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32]
        labels  = [f"E{i+1}" for i in range(5)] + ["BN"] + [f"D{i+1}" for i in range(5)]
        colors  = [enc_color]*5 + [bot_color] + [dec_color]*5 if True else []
        enc_color2, dec_color2, bot_color2 = "#00B4D8", "#7c3aed", "#f59e0b"
        clrs = [enc_color2]*5 + [bot_color2] + [dec_color2]*5

        fig_f = go.Figure(go.Bar(
            x=labels, y=filters,
            marker=dict(color=clrs, line=dict(width=0)),
            text=filters, textposition="outside",
            textfont=dict(size=10, color="#CDD6F4"),
        ))
        fig_f.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892a4"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#CDD6F4", size=10)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#6e7a8a")),
            margin=dict(l=10, r=10, t=20, b=10), height=200,
        )
        st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    section_header("ℹ️", "About This Project")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.05rem;font-weight:600;color:#CDD6F4;margin-bottom:0.7rem;">🛰️ Project Overview</div>
            <p style="color:#8892a4;font-size:0.88rem;line-height:1.7;">
                A <strong style="color:#CDD6F4;">Keras-based Deep U-Net</strong> for multi-class semantic segmentation
                of high-resolution satellite imagery. The model processes 8-band multispectral images
                to classify land-use features at the pixel level.
            </p>
            <p style="color:#8892a4;font-size:0.88rem;line-height:1.7;">
                Test-time augmentation (TTA) with <strong style="color:#CDD6F4;">7 geometric variants</strong>
                (flips, transposes, rotations) is applied at inference time and predictions are averaged,
                improving robustness and boundary accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:1rem;">
            <div style="font-size:1.05rem;font-weight:600;color:#CDD6F4;margin-bottom:0.8rem;">🗂️ SpaceNet Dataset</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
                <tr>
                    <td style="padding:6px 10px;color:#6e7a8a;border-bottom:1px solid rgba(255,255,255,0.06);">Source</td>
                    <td style="padding:6px 10px;color:#CDD6F4;border-bottom:1px solid rgba(255,255,255,0.06);">SpaceNet (commercial-grade)</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#6e7a8a;border-bottom:1px solid rgba(255,255,255,0.06);">Locations</td>
                    <td style="padding:6px 10px;color:#CDD6F4;border-bottom:1px solid rgba(255,255,255,0.06);">24 geographic scenes</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#6e7a8a;border-bottom:1px solid rgba(255,255,255,0.06);">Image type</td>
                    <td style="padding:6px 10px;color:#CDD6F4;border-bottom:1px solid rgba(255,255,255,0.06);">8-channel multispectral TIFF</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#6e7a8a;border-bottom:1px solid rgba(255,255,255,0.06);">Image depth</td>
                    <td style="padding:6px 10px;color:#CDD6F4;border-bottom:1px solid rgba(255,255,255,0.06);">16-bit per channel</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#6e7a8a;border-bottom:1px solid rgba(255,255,255,0.06);">Mask depth</td>
                    <td style="padding:6px 10px;color:#CDD6F4;border-bottom:1px solid rgba(255,255,255,0.06);">8-bit per class channel</td>
                </tr>
                <tr>
                    <td style="padding:6px 10px;color:#6e7a8a;">Training split</td>
                    <td style="padding:6px 10px;color:#CDD6F4;">75% train / 25% validation</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.05rem;font-weight:600;color:#CDD6F4;margin-bottom:0.8rem;">🚀 Getting Started</div>
            <div style="font-size:0.82rem;color:#8892a4;margin-bottom:0.5rem;">Install dependencies</div>
        </div>
        """, unsafe_allow_html=True)
        st.code("pip install -r requirements.txt", language="bash")
        st.code("python train_unet.py", language="bash")
        st.code("python predict.py", language="bash")
        st.code("streamlit run app.py", language="bash")

        st.markdown("""
        <div class="glass-card" style="margin-top:1rem;">
            <div style="font-size:1.05rem;font-weight:600;color:#CDD6F4;margin-bottom:0.8rem;">📁 Project Structure</div>
            <pre style="font-size:0.78rem;color:#4dab4d;margin:0;background:transparent;line-height:1.8;">
UNET-Segmentation/
├── app.py               ← this UI
├── train_unet.py        ← training pipeline
├── predict.py           ← inference + TTA
├── unet_model.py        ← model definition
├── gen_patches.py       ← patch sampling + augmentation
├── log_unet.csv         ← training history
├── data/
│   ├── mband/           ← 8-band input images
│   └── gt_mband/        ← segmentation masks
└── weights/
    └── unet_weights.hdf5
            </pre>
        </div>
        """, unsafe_allow_html=True)

    section_header("🌈", "Spectral Band Details")
    band_table_html = """
    <div class="glass-card">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <thead>
            <tr style="border-bottom:1px solid rgba(0,180,216,0.3);">
                <th style="padding:8px 12px;color:#6e7a8a;text-align:left;font-weight:500;">Band</th>
                <th style="padding:8px 12px;color:#6e7a8a;text-align:left;font-weight:500;">Name</th>
                <th style="padding:8px 12px;color:#6e7a8a;text-align:left;font-weight:500;">Wavelength</th>
                <th style="padding:8px 12px;color:#6e7a8a;text-align:left;font-weight:500;">Primary Use</th>
            </tr>
        </thead>
        <tbody>
    """
    uses = [
        "Atmospheric scattering, water depth",
        "Water bodies, chlorophyll absorption",
        "Vegetation health, land cover",
        "Plant stress, soil differentiation",
        "Vegetation vs bare soil",
        "Chlorophyll transition, plant stress",
        "Biomass estimation, canopy studies",
        "Vegetation moisture, crop monitoring",
    ]
    for i, ((name, wl, color), use) in enumerate(zip(BANDS, uses)):
        bg = "rgba(255,255,255,0.03)" if i % 2 == 0 else "transparent"
        band_table_html += f"""
            <tr style="background:{bg};border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:7px 12px;color:#CDD6F4;font-family:monospace;">{i+1}</td>
                <td style="padding:7px 12px;"><span style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2);
                    border:1px solid {color};border-radius:6px;padding:2px 8px;color:{color};font-size:0.8rem;">{name}</span></td>
                <td style="padding:7px 12px;color:#8892a4;font-family:monospace;">{wl}</td>
                <td style="padding:7px 12px;color:#8892a4;font-size:0.83rem;">{use}</td>
            </tr>"""
    band_table_html += "</tbody></table></div>"
    st.markdown(band_table_html, unsafe_allow_html=True)


# ── Sidebar + routing ──────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <div style="font-size:2.5rem;">🛰️</div>
            <div style="font-size:1rem;font-weight:700;color:#00B4D8;margin-top:0.3rem;">Deep UNet</div>
            <div style="font-size:0.75rem;color:#4a5568;margin-top:2px;">Satellite Segmentation</div>
        </div>
        <hr style="border-color:rgba(0,180,216,0.15);margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["Dashboard", "Data Explorer", "Inference", "Architecture", "About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(0,180,216,0.1);margin:1.2rem 0 0.8rem;'>", unsafe_allow_html=True)

        # Model status
        weights_exist = os.path.exists(
            os.path.join(os.path.dirname(__file__), "weights", "unet_weights.hdf5")
        )
        if weights_exist:
            st.markdown('<div class="status-pill status-ok" style="margin:4px 0;display:flex;width:fit-content;"><span class="status-dot"></span>Weights loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-warn" style="margin:4px 0;display:flex;width:fit-content;"><span class="status-dot"></span>No weights</div>', unsafe_allow_html=True)

        # Data status
        data_exists = os.path.exists(os.path.join(DATA_DIR, "mband", "01.tif"))
        if data_exists:
            st.markdown('<div class="status-pill status-ok" style="margin:4px 0;display:flex;width:fit-content;"><span class="status-dot"></span>Data present</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-warn" style="margin:4px 0;display:flex;width:fit-content;"><span class="status-dot"></span>Data missing</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="position:absolute;bottom:1.5rem;left:0;right:0;text-align:center;font-size:0.72rem;color:#2d3748;">
            Keras · SpaceNet · Tesla P100
        </div>
        """, unsafe_allow_html=True)

    return page


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    page = sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Data Explorer":
        page_data_explorer()
    elif page == "Inference":
        page_inference()
    elif page == "Architecture":
        page_architecture()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
