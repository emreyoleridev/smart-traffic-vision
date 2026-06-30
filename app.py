import os
import io
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Configuration
HF_REPO_ID = "emreyoleridev/smart-traffic-vision-yolov8"
HF_MODEL_FILENAME = "best.pt"
DEFAULT_CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

st.set_page_config(
    page_title="Smart Traffic Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Downloads the model from Hugging Face (if not cached) and loads it.
    Uses st.cache_resource to ensure this happens only once.
    """
    try:
        # Download model from huggingface
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            cache_dir="models"
        )
        # Load with Ultralytics
        model = YOLO(model_path)
        
        # Warmup
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy_image, verbose=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def predict(model, image: np.ndarray, conf_threshold: float):
    """Runs inference on the provided image array."""
    start_time = time.time()
    
    # Run inference
    results = model.predict(
        source=image, 
        conf=conf_threshold, 
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    inference_time_ms = (time.time() - start_time) * 1000
    detections = []
    
    result = results[0]  # We only passed one image
    names = result.names
    
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = names[class_id].lower()
        confidence = float(box.conf[0].item())
            
        xyxy = box.xyxy[0].tolist()
        
        detections.append({
            "class_name": class_name,
            "class_id": class_id,
            "confidence": confidence,
            "box": {
                "x_min": xyxy[0],
                "y_min": xyxy[1],
                "x_max": xyxy[2],
                "y_max": xyxy[3]
            }
        })
        
    return detections, inference_time_ms

def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes on the image based on detections."""
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
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=DEFAULT_CONF_THRESHOLD, step=0.05
    )
    
    st.sidebar.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
        
    if model is not None:
        st.sidebar.success("✅ Model Loaded")
    else:
        st.sidebar.error("❌ Model Load Failed")

    # Main area
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Detection 🚀", type="primary"):
            with st.spinner("Analyzing image..."):
                start_req = time.time()
                
                # Convert PIL image to numpy array for YOLO
                img_array = np.array(image)
                
                # Predict
                detections, inference_time = predict(model, img_array, conf_threshold)
                req_time = time.time() - start_req
                
                if len(detections) > 0:
                    st.success(f"Detected {len(detections)} objects!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Annotated Image")
                        annotated_img = draw_boxes(image, detections)
                        st.image(annotated_img, use_column_width=True)
                        
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
                        st.dataframe(df)
                        
                        st.markdown("### Metrics")
                        st.metric("Inference Time", f"{inference_time:.1f} ms")
                        st.metric("Total Processing Time", f"{req_time*1000:.1f} ms")
                else:
                    st.info("No objects detected with the current confidence threshold.")

if __name__ == "__main__":
    main()
