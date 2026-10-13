import streamlit as st
from PIL import Image
from predict import predict_image

st.set_page_config(page_title="Deep-Guard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: radial-gradient(circle at 20% 0%, #1b2140, #0b0e1a 60%); color: #e8ebf5;}
.block-container {max-width: 1000px; padding-top: 2.5rem;}
.brand {font-size: 28px; font-weight: 700; letter-spacing: .5px;}
.brand span {color: #7c83ff;}
.tag {color: #8b93b5; margin-bottom: 28px;}
.card {background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
       border-radius: 20px; padding: 28px; text-align: center; backdrop-filter: blur(8px);}
.ring {width: 180px; height: 180px; border-radius: 50%; margin: 8px auto 18px;
       display: flex; align-items: center; justify-content: center;}
.ring-inner {width: 140px; height: 140px; border-radius: 50%; background: #0f1322;
             display: flex; flex-direction: column; align-items: center; justify-content: center;}
.pct {font-size: 34px; font-weight: 700;}
.sub {font-size: 12px; color: #8b93b5;}
.verdict {font-size: 22px; font-weight: 600; margin-bottom: 6px;}
.hint {color: #8b93b5; font-size: 13px;}
[data-testid="stFileUploader"] section {background: rgba(255,255,255,.03);
       border: 1.5px dashed #3a4170; border-radius: 16px;}
</style>
<div class="brand">🛡️ Deep<span>Guard</span></div>
<div class="tag">Is this face real or AI-generated?</div>
""", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)

with right:
    if not file:
        st.markdown('<div class="card"><div style="font-size:48px">🔍</div>'
                    '<div class="verdict">Waiting for an image</div>'
                    '<div class="hint">Drop a face photo on the left.<br>Results appear instantly.</div></div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Scanning..."):
            label, conf = predict_image(image)

        if conf < 70:
            color, title, note = "#f5b93d", "Uncertain", "The model isn't confident. Try a clearer, front-facing photo."
        elif label == "Fake":
            color, title, note = "#ff5d6c", "Likely AI-generated", "Signs of synthetic generation detected."
        else:
            color, title, note = "#2ee59d", "Likely real", "No strong signs of manipulation found."

        st.markdown(f"""
        <div class="card">
          <div class="ring" style="background: conic-gradient({color} {conf*3.6}deg, #232846 0deg);">
            <div class="ring-inner"><div class="pct" style="color:{color}">{conf:.0f}%</div>
            <div class="sub">confidence</div></div>
          </div>
          <div class="verdict" style="color:{color}">{title}</div>
          <div class="hint">{note}</div>
        </div>""", unsafe_allow_html=True)

st.caption("Face images only · Predictions are estimates, not proof · DenseNet121 + TensorFlow")