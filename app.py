import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="YOLO vs YOLO-Pose Comparison",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 YOLO vs YOLO-Pose Vape Detection Comparison")
st.markdown("**Thesis Demo**: Comparing YOLOv8 base model with YOLOv8-Pose for improved vape detection accuracy")

# Add thesis context
with st.expander("📋 About This Demo"):
    st.markdown("""
    **Research Objective**: Demonstrate that pose estimation enhances object detection performance, 
    particularly for small handheld objects like vape devices in occluded scenarios.
    
    **Models Compared**:
    - **YOLOv8 Base**: Standard object detection with bounding boxes only
    - **YOLOv8-Pose**: Enhanced with human pose estimation for context-aware detection
    
    **Expected Benefits of Pose Model**:
    - ✅ Better accuracy in crowded scenes
    - ✅ Reduced false positives when objects are partially occluded
    - ✅ Improved detection of small handheld devices
    - ✅ Context-aware detection using human pose information
    """)

# Load models with caching
@st.cache_resource
def load_models():
    """Load both YOLO models with caching for performance"""
    try:
        # Load base YOLO model
        base_model = YOLO('vape_v2_bounding_box.pt')
        
        # Load pose YOLO model  
        pose_model = YOLO('vape_v2_pose_keypoint.pt')
        
        return base_model, pose_model
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def draw_skeleton(image, keypoints, confidence_threshold=0.5):
    """Draw pose skeleton on image"""
    # COCO pose connections (17 keypoints)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            cv2.circle(image, (int(x), int(y)), 3, colors[i % len(colors)], -1)
    
    # Draw skeleton connections
    for connection in connections:
        if (len(keypoints) > connection[0] and len(keypoints) > connection[1] and
            keypoints[connection[0]][2] > confidence_threshold and
            keypoints[connection[1]][2] > confidence_threshold):
            
            pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
            pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
            cv2.line(image, pt1, pt2, (0, 255, 255), 2)
    
    return image

def process_image(image, model, is_pose=False):
    """Process image through YOLO model and return annotated result"""
    # Run inference
    results = model(image)
    
    # Create annotated image
    annotated_image = image.copy()
    
    if len(results) > 0:
        result = results[0]
        
        # Draw bounding boxes
        if result.boxes is not None:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence text
                label = f"Vape: {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw pose keypoints if it's the pose model
        if is_pose and result.keypoints is not None:
            for keypoints in result.keypoints.xy:
                if len(keypoints) > 0:
                    # Convert to (x, y, confidence) format
                    kpts = []
                    confs = result.keypoints.conf[0] if result.keypoints.conf is not None else [1.0] * len(keypoints)
                    for i, (x, y) in enumerate(keypoints):
                        kpts.append([x.cpu().numpy(), y.cpu().numpy(), confs[i].cpu().numpy() if i < len(confs) else 1.0])
                    
                    annotated_image = draw_skeleton(annotated_image, kpts)
    
    return annotated_image, results

# Load models
with st.spinner("Loading YOLO models..."):
    base_model, pose_model = load_models()

if base_model is None or pose_model is None:
    st.error("Failed to load models. Please check the .pt files are in the correct directory.")
    st.stop()

st.success("🎉 Both models loaded successfully!")

# Presentation mode toggle
with st.sidebar:
    st.header("🎯 Demo Settings")
    presentation_mode = st.checkbox("📊 Presentation Mode", help="Optimized for thesis presentation")
    if presentation_mode:
        st.info("📋 Presentation mode active - optimized for thesis demo")
    
    st.markdown("---")
    st.markdown("**Quick Tips:**")
    st.markdown("- Upload sample videos for comparison")
    st.markdown("- Adjust frame skip for performance")
    st.markdown("- Focus on pose model advantages")
    st.markdown("- Use charts to show improvements")

# Input options
st.header("📁 Choose Input Source")
input_type = st.radio(
    "Select input type:",
    ["📁 Upload File", "📹 Live Camera"],
    horizontal=True
)

if input_type == "📁 Upload File":
    uploaded_file = st.file_uploader(
        "Choose an image or video file", 
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Upload an image or video file for detection comparison"
    )
    
    process_uploaded_file = uploaded_file
else:
    st.subheader("📹 Live Camera Demo")
    st.info("🚧 **Note**: Live camera requires additional setup. For now, you can test with uploaded videos that simulate live camera scenarios.")
    
    # For thesis demo, we'll simulate live camera with uploaded files
    st.markdown("**Simulate live camera with a video file:**")
    uploaded_file = st.file_uploader(
        "Upload a video to simulate live camera", 
        type=['mp4', 'avi', 'mov'],
        help="This will process the video as if it's coming from a live camera feed"
    )
    
    process_uploaded_file = uploaded_file

if process_uploaded_file is not None:
    file_type = process_uploaded_file.type
    
    if file_type.startswith('image'):
        # Process image
        st.subheader("🖼️ Image Processing Results")
        
        # Load and display original image
        image = Image.open(process_uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # Process with both models
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Original Image**")
            st.image(image, use_column_width=True)
        
        with col2:
            st.write("**YOLOv8 Base Model**")
            with st.spinner("Processing with base model..."):
                base_result, base_inference = process_image(image_bgr, base_model, is_pose=False)
                base_result_rgb = cv2.cvtColor(base_result, cv2.COLOR_BGR2RGB)
                st.image(base_result_rgb, use_column_width=True)
                
                # Show detection count
                detection_count = len(base_inference[0].boxes) if base_inference[0].boxes is not None else 0
                st.metric("Detections", detection_count)
        
        with col3:
            st.write("**YOLOv8-Pose Model**")
            with st.spinner("Processing with pose model..."):
                pose_result, pose_inference = process_image(image_bgr, pose_model, is_pose=True)
                pose_result_rgb = cv2.cvtColor(pose_result, cv2.COLOR_BGR2RGB)
                st.image(pose_result_rgb, use_column_width=True)
                
                # Show detection count
                detection_count = len(pose_inference[0].boxes) if pose_inference[0].boxes is not None else 0
                keypoint_count = len(pose_inference[0].keypoints.xy[0]) if pose_inference[0].keypoints is not None else 0
                st.metric("Detections", detection_count)
                st.metric("Keypoints", keypoint_count)
    
    elif file_type.startswith('video'):
        st.subheader("🎥 Video Processing Results")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(process_uploaded_file.read())
            video_path = tmp_file.name
        
        # Video processing controls
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("⚙️ Controls")
            frame_skip = st.slider("Frame Skip", 1, 10, 3, help="Process every Nth frame for performance")
            max_frames = st.slider("Max Frames", 10, 100, 30, help="Maximum frames to process")
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, help="Detection confidence threshold")
        
        with col1:
            if st.button("🎬 Process Video"):
                # Initialize video capture
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                st.info(f"Video: {total_frames} frames @ {fps:.1f} FPS")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Results containers
                results_container = st.container()
                
                # Statistics tracking
                base_detections = []
                pose_detections = []
                frame_times = []
                
                with results_container:
                    # Create columns for side-by-side video display
                    col_base, col_pose = st.columns(2)
                    
                    with col_base:
                        st.write("**YOLOv8 Base Model**")
                        base_placeholder = st.empty()
                        base_metrics = st.empty()
                    
                    with col_pose:
                        st.write("**YOLOv8-Pose Model**")
                        pose_placeholder = st.empty()
                        pose_metrics = st.empty()
                
                # Process video frames
                frame_count = 0
                processed_frames = 0
                
                while cap.isOpened() and processed_frames < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames for performance
                    if frame_count % frame_skip != 0:
                        frame_count += 1
                        continue
                    
                    start_time = time.time()
                    
                    # Process with both models
                    base_result, base_inf = process_image(frame, base_model, is_pose=False)
                    pose_result, pose_inf = process_image(frame, pose_model, is_pose=True)
                    
                    processing_time = time.time() - start_time
                    frame_times.append(processing_time)
                    
                    # Count detections
                    base_count = len(base_inf[0].boxes) if base_inf[0].boxes is not None else 0
                    pose_count = len(pose_inf[0].boxes) if pose_inf[0].boxes is not None else 0
                    
                    base_detections.append(base_count)
                    pose_detections.append(pose_count)
                    
                    # Convert BGR to RGB for display
                    base_rgb = cv2.cvtColor(base_result, cv2.COLOR_BGR2RGB)
                    pose_rgb = cv2.cvtColor(pose_result, cv2.COLOR_BGR2RGB)
                    
                    # Update display
                    with base_placeholder:
                        st.image(base_rgb, use_column_width=True)
                    with pose_placeholder:
                        st.image(pose_rgb, use_column_width=True)
                    
                    # Update metrics
                    with base_metrics:
                        st.metric("Frame Detections", base_count)
                        st.metric("Total Detections", sum(base_detections))
                    
                    with pose_metrics:
                        st.metric("Frame Detections", pose_count)
                        st.metric("Total Detections", sum(pose_detections))
                    
                    # Update progress
                    processed_frames += 1
                    progress = processed_frames / max_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {processed_frames}/{max_frames} - {processing_time:.2f}s per frame")
                    
                    frame_count += 1
                
                cap.release()
                os.unlink(video_path)  # Clean up temp file
                
                # Final comparison statistics
                st.subheader("📊 Comparison Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Base Model Total",
                        sum(base_detections),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Pose Model Total", 
                        sum(pose_detections),
                        delta=sum(pose_detections) - sum(base_detections)
                    )
                
                with col3:
                    avg_processing_time = np.mean(frame_times) if frame_times else 0
                    st.metric(
                        "Avg Processing Time",
                        f"{avg_processing_time:.2f}s"
                    )
                
                with col4:
                    effective_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    st.metric(
                        "Effective FPS",
                        f"{effective_fps:.1f}"
                    )
                
                # Performance comparison chart
                if len(base_detections) > 0:
                    st.subheader("📈 Detection Count per Frame")
                    import pandas as pd
                    
                    df = pd.DataFrame({
                        'Frame': range(len(base_detections)),
                        'Base Model': base_detections,
                        'Pose Model': pose_detections
                    })
                    
                    st.line_chart(df.set_index('Frame'))
        
else:
    st.info("👆 Please upload an image or video file to start the comparison demo")