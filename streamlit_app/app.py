import os
import io
import time
import requests
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Smart Traffic Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes on the image based on API detections."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = int(box["x_min"]), int(box["y_min"]), int(box["x_max"]), int(box["y_max"])
        conf = det["confidence"]
        cls_name = det["class_name"]
        
        # Draw bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{cls_name.capitalize()} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_cv, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def main():
    st.title("🚦 Smart Traffic Vision")
    st.markdown("""
        **End-to-End Computer Vision System for Traffic Object Detection.**
        Upload a street or traffic image to detect vehicles, pedestrians, and traffic signs.
    """)

    # Sidebar settings
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Backend API Status:")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            st.sidebar.success("✅ Online")
        else:
            st.sidebar.warning("⚠️ Degraded")
    except requests.exceptions.RequestException:
        st.sidebar.error("❌ Offline")

    # Main area
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Detection 🚀", type="primary"):
            with st.spinner("Analyzing image..."):
                # Prepare payload
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                
                files = {"file": (uploaded_file.name, img_byte_arr, "image/jpeg")}
                data = {"confidence_threshold": conf_threshold}
                
                start_req = time.time()
                try:
                    response = requests.post(f"{API_URL}/predict", files=files, data=data)
                    req_time = time.time() - start_req
                    
                    if response.status_code == 200:
                        result = response.json()
                        detections = result["detections"]
                        inference_time = result["inference_time_ms"]
                        
                        if len(detections) > 0:
                            st.success(f"Detected {len(detections)} objects!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Annotated Image")
                                annotated_img = draw_boxes(image, detections)
                                st.image(annotated_img, use_container_width=True)
                                
                                # Download button
                                buf = io.BytesIO()
                                annotated_img.save(buf, format="JPEG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="Download Annotated Image",
                                    data=byte_im,
                                    file_name="annotated_traffic.jpg",
                                    mime="image/jpeg",
                                )
                                
                            with col2:
                                st.markdown("### Detection Results")
                                # Create a dataframe for neat display
                                df = pd.DataFrame([
                                    {
                                        "Class": d["class_name"].capitalize(),
                                        "Confidence": f"{d['confidence']:.2%}",
                                        "BBox": f"[{int(d['box']['x_min'])}, {int(d['box']['y_min'])}, {int(d['box']['x_max'])}, {int(d['box']['y_max'])}]"
                                    }
                                    for d in detections
                                ])
                                st.dataframe(df, use_container_width=True)
                                
                                st.markdown("### Metrics")
                                st.metric("Inference Time", f"{inference_time:.1f} ms")
                                st.metric("Total Roundtrip Time", f"{req_time*1000:.1f} ms")
                        else:
                            st.info("No objects detected with the current confidence threshold.")
                    else:
                        st.error(f"Error from API: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    main()
