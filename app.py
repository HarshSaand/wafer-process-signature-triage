"""Streamlit process-engineer workbench for the WPST wafer-map prototype."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wafer_tcad.features import spatial_features  # noqa: E402
from wafer_tcad.metrics import nearest_neighbors  # noqa: E402
from wafer_tcad.model import WaferFusionCNN  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "model.pt"


def load_wafer_image(source: Any) -> Image.Image:
    """Load an uploaded/path-like image as an independent RGB wafer rendering."""
    image = Image.open(source).convert("RGB")
    image.load()
    return image


def image_tensor(image: Image.Image, size: int = 64) -> torch.Tensor:
    array = np.asarray(image.convert("L").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).view(1, 1, size, size)


@st.cache_resource(show_spinner=False)
def load_checkpoint(path: str) -> tuple[WaferFusionCNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "classes", "feature_cols", "config"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing: {', '.join(sorted(missing))}")
    model = WaferFusionCNN(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_image(image: Image.Image, model: WaferFusionCNN,
                  checkpoint: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Run calibrated classification, descriptors, and optional case retrieval."""
    feature_map = spatial_features(np.asarray(image))
    feature_cols = checkpoint["feature_cols"]
    values = [feature_map[name.removeprefix("feat_")] for name in feature_cols]
    features = torch.tensor([values], dtype=torch.float32)
    x = image_tensor(image)
    temperature = max(float(checkpoint.get("temperature", 1.0)), 1e-6)
    with torch.no_grad():
        logits = model(x, features) / temperature
        probabilities = torch.softmax(logits, dim=1)[0]
        embedding = model.embed(x, features)[0].numpy()
    order = torch.argsort(probabilities, descending=True)
    ranking = [
        {"pattern": checkpoint["classes"][int(index)],
         "probability": float(probabilities[index])}
        for index in order
    ]
    result: dict[str, Any] = {
        "prediction": ranking[0]["pattern"],
        "confidence": ranking[0]["probability"],
        "ranking": ranking,
        "features": feature_map,
    }
    if "reference_embeddings" in checkpoint:
        cases = nearest_neighbors(
            embedding,
            checkpoint["reference_embeddings"].numpy(),
            checkpoint["reference_labels"],
            top_k,
        )
        for case in cases:
            case["path"] = checkpoint["reference_paths"][case["index"]]
        result["similar_cases"] = cases
    return result


def _example_paths() -> list[Path]:
    root = ROOT / "data" / "raw" / "WM811k_Dataset"
    return sorted(root.glob("*/*.*")) if root.exists() else []


def _render_header() -> None:
    st.markdown(
        """<div class="hero"><span class="eyebrow">WAFER PROCESS SIGNATURE TRIAGE</span>
        <h1>WPST Engineer Workbench</h1>
        <p>Calibrated wafer-map pattern screening with polar-spatial signatures and
        nearest known cases.</p></div>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="WPST · Wafer Triage", page_icon="◉", layout="wide")
    st.markdown("""<style>
      :root {--navy:#071c2c;--cyan:#13b8c7;--amber:#ffb547;--mist:#edf5f7}
      .stApp {background:linear-gradient(145deg,#f7fbfc 0%,#eaf3f5 100%);color:var(--navy)}
      .hero {padding:1.5rem 1.8rem;border-left:7px solid var(--cyan);background:var(--navy);
             color:white;border-radius:4px;margin-bottom:1rem;box-shadow:0 8px 28px #071c2c22}
      .hero h1 {margin:.2rem 0;font-weight:650;letter-spacing:-.03em}.hero p{margin:0;color:#c9e0e5}
      .eyebrow {font-size:.72rem;letter-spacing:.18em;color:#65d7df;font-weight:700}
      [data-testid="stMetric"] {background:white;border-top:3px solid var(--cyan);padding:.8rem}
      .review {border-left:5px solid var(--amber);background:#fff8e9;padding:.7rem 1rem;border-radius:3px}
    </style>""", unsafe_allow_html=True)
    _render_header()

    with st.sidebar:
        st.header("Run controls")
        checkpoint_path = st.text_input("Checkpoint", str(DEFAULT_CHECKPOINT))
        review_threshold = st.slider(
            "Human-review threshold", 0.0, 1.0, 0.65, 0.05,
            help="Operational scenario control—not a threshold validated by this study.",
        )
        top_k = st.slider("Similar cases", 1, 10, 5)
        st.caption("Prototype decision support only. A predicted pattern is not a process root cause or TCAD result.")

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        st.warning("No trained checkpoint found. Run `python scripts/train.py`, then refresh this page.")
        st.code("python scripts/train.py\nstreamlit run app.py", language="bash")
        return
    try:
        model, metadata = load_checkpoint(str(checkpoint.resolve()))
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return

    source = st.radio("Input source", ["Upload image", "Curated dataset example"], horizontal=True)
    image: Image.Image | None = None
    source_name = ""
    if source == "Upload image":
        upload = st.file_uploader("Wafer map image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if upload:
            image, source_name = load_wafer_image(io.BytesIO(upload.getvalue())), upload.name
    else:
        examples = _example_paths()
        if not examples:
            st.info("Prepare the dataset first to enable curated examples.")
        else:
            labels = [str(path.relative_to(ROOT)) for path in examples]
            chosen = st.selectbox("Dataset sample", labels)
            image, source_name = load_wafer_image(ROOT / chosen), chosen

    if image is None:
        st.info("Choose a wafer map to begin triage.")
        return

    with st.spinner("Extracting spatial signature and running calibrated inference…"):
        try:
            result = predict_image(image, model, metadata, top_k)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    image_col, result_col = st.columns([1, 1.7], gap="large")
    with image_col:
        st.subheader("Wafer input")
        st.image(image, caption=source_name, use_container_width=True)
        st.caption(f"Original dimensions: {image.width} × {image.height} px · inference resize: 64 × 64")
    with result_col:
        st.subheader("Triage output")
        a, b, c = st.columns(3)
        a.metric("Predicted signature", result["prediction"])
        b.metric("Calibrated confidence", f"{result['confidence']:.1%}")
        c.metric("Routing", "Review" if result["confidence"] < review_threshold else "Screened")
        if result["confidence"] < review_threshold:
            st.markdown("<div class='review'><b>Human review recommended.</b> Confidence is below the selected operational scenario threshold.</div>", unsafe_allow_html=True)
        chart = pd.DataFrame(result["ranking"][:5]).set_index("pattern")
        st.bar_chart(chart, color="#13b8c7", horizontal=True)

    st.subheader("Engineer-readable spatial signature")
    preferred = ["defect_fraction", "radial_mean", "radial_std", "core_density",
                 "middle_density", "edge_density", "anisotropy", "angular_entropy"]
    signature = pd.DataFrame({"descriptor": preferred,
                              "value": [result["features"][key] for key in preferred]})
    st.dataframe(signature, hide_index=True, use_container_width=True,
                 column_config={"value": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")})
    st.caption("Descriptors summarize geometry for triage and comparison; they do not identify a physical mechanism by themselves.")

    if result.get("similar_cases"):
        st.subheader("Nearest known training cases")
        cols = st.columns(min(len(result["similar_cases"]), 5))
        for col, case in zip(cols, result["similar_cases"]):
            candidate = Path(case["path"])
            candidate = candidate if candidate.is_absolute() else ROOT / candidate
            with col:
                if candidate.exists():
                    st.image(str(candidate), use_container_width=True)
                st.markdown(f"**{case['label']}**")
                st.caption(f"Cosine similarity: {case['similarity']:.3f}")

    with st.expander("How this complements a process/TCAD workflow"):
        st.markdown("""
        1. Screen wafer-test signatures and prioritize uncertain or unusual cases.
        2. Compare a case with known signatures and spatial descriptors.
        3. Join flagged wafers to lot, tool, recipe, metrology, and sensor context.
        4. Form a process hypothesis; use calibrated TCAD or designed experiments to test it.

        WPST performs steps 1–2. It does **not** simulate fabrication physics, infer causality,
        or replace engineer review.
        """)


if __name__ == "__main__":
    main()
