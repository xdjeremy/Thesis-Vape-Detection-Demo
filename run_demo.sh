#!/bin/bash

# YOLO vs YOLO-Pose Demo Startup Script
# For Thesis Presentation

echo "🔍 Starting YOLO vs YOLO-Pose Comparison Demo..."
echo "📋 Thesis Demo: Pose Estimation for Enhanced Vape Detection"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/lib/python*/site-packages/streamlit/__init__.py" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if models exist
if [ ! -f "vape_v2_bounding_box.pt" ] || [ ! -f "vape_v2_pose_keypoint.pt" ]; then
    echo "❌ Model files not found!"
    echo "Please ensure these files are in the current directory:"
    echo "  - vape_v2_bounding_box.pt"
    echo "  - vape_v2_pose_keypoint.pt"
    exit 1
fi

echo "✅ Models found:"
echo "  - vape_v2_bounding_box.pt"
echo "  - vape_v2_pose_keypoint.pt"
echo ""

echo "🚀 Starting Streamlit demo..."
echo "📱 Demo will open in your browser at: http://localhost:8501"
echo ""
echo "📋 For your thesis presentation:"
echo "  1. Upload your sample videos"
echo "  2. Compare detection results side-by-side"
echo "  3. Show improved performance with pose estimation"
echo ""

# Start Streamlit
streamlit run app.py