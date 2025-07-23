# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based thesis demo application comparing YOLOv8 base model with YOLOv8-Pose model for vape device detection. The application demonstrates how pose estimation can improve object detection accuracy, particularly for small handheld objects in occluded scenarios.

## Essential Commands

### Setup and Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Quick setup using convenience script
./run_demo.sh
```

### Running the Application
```bash
# Start the Streamlit demo
streamlit run app.py

# Test model loading
python test_models.py
```

### Model Requirements
The application requires two pre-trained model files in the root directory:
- `vape_v2_bounding_box.pt` - YOLOv8 base model for vape detection
- `vape_v2_pose_keypoint.pt` - YOLOv8-Pose model with pose estimation

## Core Architecture

### Application Structure
- **Single-file Streamlit app** (`app.py`) with comprehensive demo interface
- **Model caching** using `@st.cache_resource` for performance optimization
- **Three processing modes**: Image upload, video processing, live camera simulation
- **Real-time comparison** with side-by-side model analysis

### Key Processing Functions

#### Detection and Analysis Pipeline
- `process_image()` - Basic YOLO inference with annotation
- `process_image_enhanced()` - Advanced processing with detection data extraction
- `extract_detection_data()` - Converts YOLO results to structured detection format
- `match_detections()` - IoU-based detection matching between models
- `calculate_iou()` - Intersection over Union calculation for bounding box comparison

#### Visualization Components
- `draw_skeleton()` - Renders pose keypoints and skeletal connections (17 COCO keypoints)
- `draw_comparison_results()` - Color-coded visualization of detection differences:
  - Green: Matched detections (both models agree)
  - Red: False positives (base model only)  
  - Blue: Pose advantages (pose model only)

#### Video Processing
- Frame-by-frame analysis with configurable frame skip for performance
- Progress tracking and time estimation
- Real-time metrics display during processing
- Performance comparison charts using pandas/streamlit

### Detection Matching Algorithm
The core comparison uses IoU-based matching:
1. Calculate IoU between all base and pose model detections
2. Match detections above configurable IoU threshold (default: 0.5)
3. Classify unmatched detections as either false positives or pose advantages
4. Generate quantitative metrics for thesis presentation

### Session State Management
For live camera mode, the app maintains:
- `camera_stats` - Accumulated detection statistics
- `freeze_frame` - Captured frame data for detailed analysis
- Camera control states (active, paused, freeze)

## Model Loading Patterns

Models are loaded using Ultralytics YOLO with error handling:
```python
@st.cache_resource
def load_models():
    base_model = YOLO('vape_v2_bounding_box.pt')
    pose_model = YOLO('vape_v2_pose_keypoint.pt')
    return base_model, pose_model
```

## Performance Considerations

- **Optimized for MacBook M2** performance
- **Frame skipping** in video/live processing for real-time performance
- **Confidence thresholding** to filter low-quality detections (default: 0.3-0.5)
- **Model caching** to avoid repeated loading
- **Progress indicators** for long-running operations

## Thesis Demo Features

### Analysis Modes
- **Quantitative comparison** with IoU-based detection matching
- **False positive highlighting** showing where base model fails
- **Freeze frame analysis** for detailed examination of specific moments
- **Export functionality** for captured frames and analysis reports

### Presentation Optimizations
- Professional layout with clear visual hierarchy  
- Real-time processing capabilities
- Session statistics and performance metrics
- Color-coded analysis results for easy interpretation

## Dependencies

Core requirements from `requirements.txt`:
- `streamlit>=1.28.0` - Web application framework
- `ultralytics>=8.0.0` - YOLO model inference
- `opencv-python>=4.8.0` - Computer vision processing
- `torch>=2.0.0` - PyTorch backend
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical operations