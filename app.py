import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# st.set_page_config MUST be the first Streamlit command
# in the entire file. It controls browser tab settings.
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🕳️",
    layout="wide"
)

# @st.cache_resource means: load the model ONCE,
# and keep it in memory — don't reload it every time
# the page refreshes.
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# ── Helper functions ────────────────────────────────────────────

def get_severity(area_percent):
    """
    Classifies a pothole's severity based on how much
    of the total image area its bounding box covers.
    """
    if area_percent < 3:
        return "Small", "🟡"
    elif area_percent < 8:
        return "Medium", "🟠"
    else:
        return "Large", "🔴"

def calculate_road_health(detections_list, image_count=1):
    """
    Road Health Score that accounts for HOW MANY potholes
    were found, not just their relative size in the frame.
    """
    if not detections_list:
        return 100.0

    weights = {"Small": 8, "Medium": 18, "Large": 35}

    # Per-pothole deduction
    deductions = sum(
        weights.get(d["severity"], 12) * d["confidence"]
        for d in detections_list
    )

    # Extra penalty simply for HIGH COUNT of potholes —
    # a road with 5 potholes is worse than 1, even if
    # each individually measures "Small" by area
    count_penalty = min(len(detections_list) * 4, 25)

    score = max(0, 100 - deductions - count_penalty)
    return round(score, 1)

def get_grade(score):
    """Convert a numeric health score into a letter grade,
    condition label, and a display color."""
    if score >= 85:
        return "A", "Excellent", "#27ae60"
    elif score >= 70:
        return "B", "Good", "#2ecc71"
    elif score >= 55:
        return "C", "Fair", "#f39c12"
    elif score >= 40:
        return "D", "Poor", "#e67e22"
    else:
        return "F", "Critical", "#e74c3c"

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.write("---")
    st.markdown("### 📊 Model Performance")
    st.metric("Precision", "93.9%")
    st.metric("Recall", "82.8%")
    st.metric("mAP50", "84.0%")
    st.metric("Speed", "63 FPS")

    st.write("---")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Higher = stricter, fewer false alarms. Lower = catches more, but more false positives."
    )

    st.write("---")
    st.markdown("### 🎨 Severity Color Guide")
    st.markdown("🟡 **Small** — under 3% of image")
    st.markdown("🟠 **Medium** — 3% to 8% of image")
    st.markdown("🔴 **Large** — over 8% of image")

# ── Header + Live Metrics ─────────────────────────────────────
# ── Gradient header band (subtle background interest) ────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #3a2a1a 100%);
        padding: 28px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 4px solid #EF9F27;
    ">
        <h1 style="margin:0; color:#f0f0f0; font-size:32px;">
            🕳️ Pothole Detection System
        </h1>
        <p style="margin:6px 0 0 0; color:#bbb; font-size:14px;">
            Real-time road damage detection powered by YOLOv8
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# ── Upload + Detection ─────────────────────────────────────────
st.write("---")
st.subheader("Upload a road image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_h, img_w = img_array.shape[0], img_array.shape[1]
    img_area = img_h * img_w

    # Two equal columns side by side
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original image**")
        st.image(image, width="stretch")

    results = model.predict(image, conf=confidence_threshold)
    annotated_image = results[0].plot()
    num_detections = len(results[0].boxes)

    with col2:
        st.write("**Detection result**")
        st.image(annotated_image, width="stretch")

    st.write("---")

    if num_detections > 0:
        st.subheader(f"🕳️ {num_detections} pothole(s) detected")

        # Build a table row for each detection, and also
        # collect simplified data for the health score calc
        table_data = []
        detections_for_score = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            area_pct = (box_area / img_area) * 100
            severity_label, severity_icon = get_severity(area_pct)

            table_data.append({
                "Severity": f"{severity_icon} {severity_label}",
                "Confidence": f"{conf:.1%}",
                "Size (% of image)": f"{area_pct:.1f}%"
            })

            detections_for_score.append({
                "severity": severity_label,
                "confidence": conf
            })

        st.table(table_data)

        # ── Road Health Score ───────────────────────────────
        st.write("---")
        health_score = calculate_road_health(detections_for_score)
        grade, condition, color = get_grade(health_score)

        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.metric("🛣️ Road Health Score", f"{health_score}")
        with score_col2:
            st.markdown(
                f"<div style='background:{color};color:white;"
                f"padding:12px;border-radius:8px;text-align:center;"
                f"font-size:18px;font-weight:bold;'>"
                f"Grade {grade} — {condition}</div>",
                unsafe_allow_html=True
            )

        # ── Download annotated result ───────────────────────
        st.write("---")
        result_pil = Image.fromarray(annotated_image)
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        st.download_button(
            label="⬇️ Download Result Image",
            data=buf,
            file_name="pothole_detection_result.jpg",
            mime="image/jpeg"
        )

    else:
        st.info("No potholes detected above the current confidence threshold.")
        st.write("Try lowering the confidence threshold in the sidebar.")

else:
    st.info("👆 Upload a road image above to begin detection.")

# ── Footer ───────────────────────────────────────────────────
st.write("---")
