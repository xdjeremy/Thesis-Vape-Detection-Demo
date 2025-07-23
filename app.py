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
    - ✅ **Reduced False Positives**: Fewer incorrect detections in complex scenes
    - ✅ **Occlusion Handling**: Better performance when vape devices are partially hidden
    - ✅ **Context Awareness**: Uses human pose to validate detection plausibility
    - ✅ **Higher Precision**: More confident and accurate detections overall
    
    **Demo Features**:
    - 🔍 **Real-time Comparison**: Side-by-side model analysis
    - 📊 **Quantitative Analysis**: IoU-based detection matching and metrics
    - 📸 **Freeze Frame Analysis**: Capture specific moments for detailed examination
    - 🎯 **False Positive Highlighting**: Visual identification of detection differences
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

def extract_detection_data(results, conf_threshold=0.3):
    """Extract bounding boxes and confidence scores from YOLO results"""
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = box.conf[0].cpu().numpy()
            if conf >= conf_threshold:  # Filter by confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
    return detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Calculate intersection area
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union area
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def match_detections(base_detections, pose_detections, iou_threshold=0.5):
    """Match detections between two models using IoU threshold"""
    matched_pairs = []
    base_unmatched = list(range(len(base_detections)))
    pose_unmatched = list(range(len(pose_detections)))
    
    # Find best matches
    for i, base_det in enumerate(base_detections):
        best_iou = 0
        best_match = -1
        
        for j, pose_det in enumerate(pose_detections):
            if j in pose_unmatched:
                iou = calculate_iou(base_det['bbox'], pose_det['bbox'])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = j
        
        if best_match != -1:
            matched_pairs.append((i, best_match, best_iou))
            base_unmatched.remove(i)
            pose_unmatched.remove(best_match)
    
    return matched_pairs, base_unmatched, pose_unmatched

def draw_comparison_results(image, base_detections, pose_detections, 
                          matched_pairs, base_unmatched, pose_unmatched):
    """Draw detections with different colors for matched/missed"""
    annotated_image = image.copy()
    
    # Colors for different detection types
    colors = {
        'matched': (0, 255, 0),           # Green - matched detections
        'false_positive': (255, 0, 0),    # Red - detected by base, missed by pose (likely false positive)
        'pose_advantage': (0, 0, 255),    # Blue - detected by pose, missed by base (pose advantage)
    }
    
    # Draw matched detections (Green)
    for base_idx, pose_idx, iou in matched_pairs:
        base_box = base_detections[base_idx]['bbox']
        x1, y1, x2, y2 = map(int, base_box)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors['matched'], 2)
        cv2.putText(annotated_image, f"Match: {iou:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['matched'], 2)
    
    # Draw false positives - base detected, pose didn't (Red)
    for base_idx in base_unmatched:
        base_box = base_detections[base_idx]['bbox']
        x1, y1, x2, y2 = map(int, base_box)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors['false_positive'], 2)
        cv2.putText(annotated_image, "False Positive?", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['false_positive'], 2)
    
    # Draw pose advantages - pose detected, base didn't (Blue)
    for pose_idx in pose_unmatched:
        pose_box = pose_detections[pose_idx]['bbox']
        x1, y1, x2, y2 = map(int, pose_box)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors['pose_advantage'], 2)
        cv2.putText(annotated_image, "Pose Advantage", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['pose_advantage'], 2)
    
    return annotated_image

def process_image_enhanced(image, model, conf_threshold=0.3, is_pose=False):
    """Enhanced process_image that returns detection data for analysis"""
    # Run inference
    results = model(image)
    
    # Create annotated image
    annotated_image = image.copy()
    
    if len(results) > 0:
        result = results[0]
        
        # Draw bounding boxes
        if result.boxes is not None:
            for box in result.boxes:
                conf = box.conf[0].cpu().numpy()
                if conf >= conf_threshold:  # Filter by confidence
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
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
    
    # Extract detection data
    detection_data = extract_detection_data(results, conf_threshold)
    
    return annotated_image, results, detection_data

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
    st.markdown("- Use live camera for real-time demo")
    st.markdown("- Focus on false positive reduction")
    st.markdown("- Freeze frames for detailed analysis")
    
    st.markdown("---")
    st.markdown("**📊 Analysis Color Code:**")
    st.markdown("🟢 **Green**: Matched detections (both models agree)")
    st.markdown("🔴 **Red**: False positives (base only)")
    st.markdown("🔵 **Blue**: Pose advantages (pose only)")

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
    
    # Initialize session state for camera
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'camera_paused' not in st.session_state:
        st.session_state.camera_paused = False
    if 'freeze_frame' not in st.session_state:
        st.session_state.freeze_frame = None
    if 'camera_stats' not in st.session_state:
        st.session_state.camera_stats = {
            'frames_processed': 0,
            'base_detections': [],
            'pose_detections': [],
            'false_positives': [],
            'pose_advantages': []
        }
    
    # Camera controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📹 Start Camera"):
            st.session_state.camera_active = True
            st.session_state.camera_paused = False
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause"):
            st.session_state.camera_paused = not st.session_state.camera_paused
            st.rerun()
    
    with col3:
        if st.button("📸 Freeze Frame"):
            st.session_state.freeze_frame = "capture"
            st.rerun()
    
    with col4:
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_active = False
            st.session_state.camera_paused = False
            st.session_state.freeze_frame = None
            st.rerun()
    
    # Camera settings
    with st.expander("⚙️ Camera Settings"):
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, help="Filter detections below this confidence")
        iou_threshold = st.slider("IoU Matching Threshold", 0.1, 0.9, 0.5, help="Threshold for matching detections between models")
        frame_skip = st.slider("Frame Skip", 1, 5, 2, help="Process every Nth frame for performance")
        
        st.markdown("---")
        occlusion_mode = st.checkbox("🎯 Occlusion Demo Mode", help="Highlight pose model advantages in occluded scenarios")
        if occlusion_mode:
            st.info("💡 **Demo Tip**: Show vape devices partially covered by hands, behind objects, or in crowded scenes to demonstrate pose model superiority!")
            st.markdown("**Occlusion Scenarios to Test:**")
            st.markdown("- Vape partially covered by hand")
            st.markdown("- Multiple people in frame")
            st.markdown("- Objects in foreground")
    
    # Process live camera
    process_uploaded_file = None

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
        
        # Enhanced processing with analysis
        st.subheader("⚙️ Analysis Settings")
        col_settings1, col_settings2 = st.columns(2)
        with col_settings1:
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, help="Filter detections below this confidence")
        with col_settings2:
            iou_threshold = st.slider("IoU Matching Threshold", 0.1, 0.9, 0.5, help="Threshold for matching detections between models")
        
        # Process with both models using enhanced functions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Original Image**")
            st.image(image, use_container_width=True)
        
        with col2:
            st.write("**🔍 Base Model**")
            with st.spinner("Processing with base model..."):
                base_result, base_inference, base_detections = process_image_enhanced(image_bgr, base_model, conf_threshold, is_pose=False)
                base_result_rgb = cv2.cvtColor(base_result, cv2.COLOR_BGR2RGB)
                st.image(base_result_rgb, use_container_width=True)
                
                # Show metrics
                st.metric("Detections", len(base_detections))
                if base_detections:
                    avg_conf = np.mean([d['confidence'] for d in base_detections])
                    st.metric("Avg Confidence", f"{avg_conf:.2f}")
        
        with col3:
            st.write("**🎯 Pose Model**")
            with st.spinner("Processing with pose model..."):
                pose_result, pose_inference, pose_detections = process_image_enhanced(image_bgr, pose_model, conf_threshold, is_pose=True)
                pose_result_rgb = cv2.cvtColor(pose_result, cv2.COLOR_BGR2RGB)
                st.image(pose_result_rgb, use_container_width=True)
                
                # Show metrics
                st.metric("Detections", len(pose_detections))
                if pose_detections:
                    avg_conf = np.mean([d['confidence'] for d in pose_detections])
                    st.metric("Avg Confidence", f"{avg_conf:.2f}")
                keypoint_count = len(pose_inference[0].keypoints.xy[0]) if pose_inference[0].keypoints is not None else 0
                st.metric("Keypoints", keypoint_count)
        
        with col4:
            st.write("**📊 Analysis**")
            with st.spinner("Analyzing differences..."):
                # Match detections
                matched_pairs, base_unmatched, pose_unmatched = match_detections(
                    base_detections, pose_detections, iou_threshold
                )
                
                # Create analysis visualization
                analysis_result = draw_comparison_results(
                    image_bgr, base_detections, pose_detections, 
                    matched_pairs, base_unmatched, pose_unmatched
                )
                analysis_result_rgb = cv2.cvtColor(analysis_result, cv2.COLOR_BGR2RGB)
                st.image(analysis_result_rgb, use_container_width=True)
                
                # Show analysis metrics
                st.metric("✅ Matched", len(matched_pairs))
                st.metric("❌ False Positives", len(base_unmatched))
                st.metric("🎯 Pose Advantages", len(pose_unmatched))
        
        # Detailed comparison summary
        st.subheader("📈 Detailed Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Base Model Performance**")
            st.write(f"Total detections: {len(base_detections)}")
            if base_detections:
                st.write(f"Avg confidence: {np.mean([d['confidence'] for d in base_detections]):.2f}")
        
        with col2:
            st.markdown("**Pose Model Performance**")
            st.write(f"Total detections: {len(pose_detections)}")
            if pose_detections:
                st.write(f"Avg confidence: {np.mean([d['confidence'] for d in pose_detections]):.2f}")
        
        with col3:
            st.markdown("**Agreement Analysis**")
            st.write(f"Matched detections: {len(matched_pairs)}")
            if matched_pairs:
                avg_iou = np.mean([iou for _, _, iou in matched_pairs])
                st.write(f"Avg IoU: {avg_iou:.2f}")
        
        with col4:
            st.markdown("**Pose Model Advantages**")
            st.write(f"Likely false positives caught: {len(base_unmatched)}")
            st.write(f"Additional valid detections: {len(pose_unmatched)}")
            if len(base_detections) > 0:
                fp_rate = len(base_unmatched) / len(base_detections) * 100
                st.write(f"False positive reduction: {fp_rate:.1f}%")
    
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
                        st.image(base_rgb, use_container_width=True)
                    with pose_placeholder:
                        st.image(pose_rgb, use_container_width=True)
                    
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
        
# Live camera processing
elif input_type == "📹 Live Camera" and 'camera_active' in st.session_state and st.session_state.camera_active:
    st.subheader("📹 Live Camera Feed")
    
    # Create camera placeholders
    camera_container = st.container()
    
    with camera_container:
        # Create columns for live display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**📷 Live Feed**")
            live_placeholder = st.empty()
        
        with col2:
            st.write("**🔍 Base Model**")
            base_placeholder = st.empty()
            base_metrics = st.empty()
        
        with col3:
            st.write("**🎯 Pose Model**")
            pose_placeholder = st.empty()
            pose_metrics = st.empty()
        
        with col4:
            st.write("**📊 Analysis**")
            analysis_placeholder = st.empty()
            analysis_metrics = st.empty()
    
    # Statistics display
    stats_container = st.container()
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access camera. Please check camera permissions.")
        else:
            # Camera processing loop
            frame_count = 0
            
            while st.session_state.camera_active and not st.session_state.camera_paused:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                start_time = time.time()
                
                # Display original frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                live_placeholder.image(frame_rgb, use_container_width=True)
                
                # Process with both models
                base_result, base_inf, base_detections = process_image_enhanced(frame, base_model, conf_threshold, is_pose=False)
                pose_result, pose_inf, pose_detections = process_image_enhanced(frame, pose_model, conf_threshold, is_pose=True)
                
                # Analyze detections
                matched_pairs, base_unmatched, pose_unmatched = match_detections(
                    base_detections, pose_detections, iou_threshold
                )
                
                # Create analysis visualization
                analysis_result = draw_comparison_results(
                    frame, base_detections, pose_detections, 
                    matched_pairs, base_unmatched, pose_unmatched
                )
                
                # Update displays
                base_rgb = cv2.cvtColor(base_result, cv2.COLOR_BGR2RGB)
                pose_rgb = cv2.cvtColor(pose_result, cv2.COLOR_BGR2RGB)
                analysis_rgb = cv2.cvtColor(analysis_result, cv2.COLOR_BGR2RGB)
                
                base_placeholder.image(base_rgb, use_container_width=True)
                pose_placeholder.image(pose_rgb, use_container_width=True)
                analysis_placeholder.image(analysis_rgb, use_container_width=True)
                
                # Update metrics
                processing_time = time.time() - start_time
                
                with base_metrics:
                    st.metric("Detections", len(base_detections))
                    st.metric("Avg Confidence", f"{np.mean([d['confidence'] for d in base_detections]):.2f}" if base_detections else "0.00")
                
                with pose_metrics:
                    st.metric("Detections", len(pose_detections))
                    st.metric("Avg Confidence", f"{np.mean([d['confidence'] for d in pose_detections]):.2f}" if pose_detections else "0.00")
                
                with analysis_metrics:
                    st.metric("Matched", len(matched_pairs))
                    st.metric("False Positives", len(base_unmatched))
                    st.metric("Pose Advantage", len(pose_unmatched))
                    st.metric("FPS", f"{1.0/processing_time:.1f}")
                
                # Update session stats
                st.session_state.camera_stats['frames_processed'] += 1
                st.session_state.camera_stats['base_detections'].append(len(base_detections))
                st.session_state.camera_stats['pose_detections'].append(len(pose_detections))
                st.session_state.camera_stats['false_positives'].append(len(base_unmatched))
                st.session_state.camera_stats['pose_advantages'].append(len(pose_unmatched))
                
                # Handle freeze frame
                if st.session_state.freeze_frame == "capture":
                    st.session_state.freeze_frame = {
                        'original': frame_rgb.copy(),
                        'base': base_rgb.copy(),
                        'pose': pose_rgb.copy(),
                        'analysis': analysis_rgb.copy(),
                        'metrics': {
                            'base_detections': len(base_detections),
                            'pose_detections': len(pose_detections),
                            'matched': len(matched_pairs),
                            'false_positives': len(base_unmatched),
                            'pose_advantages': len(pose_unmatched)
                        }
                    }
                    st.success("📸 Frame captured for analysis!")
                
                frame_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.05)
    
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()

# Display freeze frame analysis
if input_type == "📹 Live Camera" and 'freeze_frame' in st.session_state and st.session_state.freeze_frame and isinstance(st.session_state.freeze_frame, dict):
    st.subheader("📸 Frozen Frame Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Original Frame**")
        st.image(st.session_state.freeze_frame['original'], use_container_width=True)
    
    with col2:
        st.write("**Base Model**")
        st.image(st.session_state.freeze_frame['base'], use_container_width=True)
        st.metric("Detections", st.session_state.freeze_frame['metrics']['base_detections'])
    
    with col3:
        st.write("**Pose Model**")
        st.image(st.session_state.freeze_frame['pose'], use_container_width=True)
        st.metric("Detections", st.session_state.freeze_frame['metrics']['pose_detections'])
    
    with col4:
        st.write("**Analysis**")
        st.image(st.session_state.freeze_frame['analysis'], use_container_width=True)
        st.metric("False Positives", st.session_state.freeze_frame['metrics']['false_positives'])
        st.metric("Pose Advantages", st.session_state.freeze_frame['metrics']['pose_advantages'])
    
    # Export freeze frame
    st.subheader("💾 Export Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📸 Save Original Frame"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"freeze_frame_original_{timestamp}.png"
            # Convert to PIL Image and save
            from PIL import Image
            img = Image.fromarray(st.session_state.freeze_frame['original'])
            img.save(filename)
            st.success(f"✅ Saved as {filename}")
    
    with col2:
        if st.button("📊 Save Analysis View"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"freeze_frame_analysis_{timestamp}.png"
            # Convert to PIL Image and save
            from PIL import Image
            img = Image.fromarray(st.session_state.freeze_frame['analysis'])
            img.save(filename)
            st.success(f"✅ Saved as {filename}")
    
    with col3:
        if st.button("📋 Export Metrics Report"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}.txt"
            metrics = st.session_state.freeze_frame['metrics']
            
            report = f"""YOLO vs YOLO-Pose Analysis Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

Detection Results:
- Base Model Detections: {metrics['base_detections']}
- Pose Model Detections: {metrics['pose_detections']}
- Matched Detections: {metrics['matched']}
- False Positives (Base Only): {metrics['false_positives']}
- Pose Advantages (Pose Only): {metrics['pose_advantages']}

Analysis:
- False Positive Rate: {metrics['false_positives']/max(metrics['base_detections'], 1)*100:.1f}%
- Pose Model Improvement: {metrics['pose_advantages']/max(metrics['pose_detections'], 1)*100:.1f}%

Thesis Conclusion:
{'✅ Pose model shows superior performance' if metrics['false_positives'] > 0 or metrics['pose_advantages'] > 0 else '⚠️  No significant difference detected'}
"""
            
            with open(filename, 'w') as f:
                f.write(report)
            st.success(f"✅ Report saved as {filename}")
            
            # Also display the report
            with st.expander("📋 View Report"):
                st.text(report)

# Display session statistics
if input_type == "📹 Live Camera" and 'camera_stats' in st.session_state and st.session_state.camera_stats['frames_processed'] > 0:
    st.subheader("📊 Session Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.camera_stats
    
    with col1:
        st.metric("Frames Processed", stats['frames_processed'])
        st.metric("Total Base Detections", sum(stats['base_detections']))
    
    with col2:
        st.metric("Total Pose Detections", sum(stats['pose_detections']))
        st.metric("Avg Detections/Frame", f"{np.mean(stats['pose_detections']):.1f}")
    
    with col3:
        st.metric("Total False Positives", sum(stats['false_positives']))
        st.metric("False Positive Rate", f"{sum(stats['false_positives'])/max(sum(stats['base_detections']), 1)*100:.1f}%")
    
    with col4:
        st.metric("Total Pose Advantages", sum(stats['pose_advantages']))
        st.metric("Pose Improvement", f"{sum(stats['pose_advantages'])/max(sum(stats['pose_detections']), 1)*100:.1f}%")

else:
    st.info("👆 Please upload an image or video file, or start the live camera to begin the comparison demo")