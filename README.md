# YOLO vs YOLO-Pose Comparison Demo

This is a Streamlit web application that demonstrates the performance comparison between YOLOv8 base model and YOLOv8-Pose model for vape device detection.

## Features

- **Side-by-side comparison** of YOLOv8 vs YOLOv8-Pose models
- **Image processing** with real-time detection visualization
- **Video processing** with frame-by-frame analysis
- **Pose keypoint visualization** with skeleton overlay
- **Performance metrics** and comparison charts
- **Live camera simulation** for presentation purposes

## Setup

1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Access the demo:**
   Open your browser to `http://localhost:8501`

## Usage

### Image Processing
1. Select "📁 Upload File"
2. Upload an image (JPG, PNG)
3. View side-by-side detection results
4. Compare detection counts and confidence scores

### Video Processing
1. Select "📁 Upload File" 
2. Upload a video (MP4, AVI, MOV)
3. Adjust processing parameters:
   - **Frame Skip**: Process every Nth frame for performance
   - **Max Frames**: Limit total frames processed
   - **Confidence Threshold**: Detection confidence threshold
4. Click "🎬 Process Video"
5. View real-time processing and final comparison statistics

### Live Camera Simulation
1. Select "📹 Live Camera"
2. Upload a video file to simulate live camera feed
3. Process as real-time demo

## Models

- `vape_v2_bounding_box.pt`: YOLOv8 base model for vape detection
- `vape_v2_pose_keypoint.pt`: YOLOv8-Pose model with pose estimation

## Performance

- Optimized for MacBook M2 performance
- Real-time processing capabilities
- Frame skipping for improved performance
- Progress tracking and time estimation

## Thesis Demo Features

- Professional presentation layout
- Clear performance metrics
- Visual comparison charts
- Pose skeleton visualization
- Processing time analysis

Perfect for demonstrating how pose estimation improves vape detection accuracy, especially in occluded scenarios.