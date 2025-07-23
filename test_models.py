#!/usr/bin/env python3
"""
Test script to verify YOLO models can be loaded properly
"""
import os
from ultralytics import YOLO

def test_model_loading():
    """Test if both YOLO models can be loaded"""
    print("🔧 Testing YOLO model loading...")
    
    # Check if model files exist
    base_model_path = 'vape_v2_bounding_box.pt'
    pose_model_path = 'vape_v2_pose_keypoint.pt'
    
    if not os.path.exists(base_model_path):
        print(f"❌ Base model not found: {base_model_path}")
        return False
        
    if not os.path.exists(pose_model_path):
        print(f"❌ Pose model not found: {pose_model_path}")
        return False
    
    try:
        # Load base YOLO model
        print("Loading base YOLO model...")
        base_model = YOLO(base_model_path)
        print("✅ Base YOLO model loaded successfully")
        print(f"   Model type: {type(base_model.model)}")
        
        # Load pose YOLO model  
        print("Loading YOLO-Pose model...")
        pose_model = YOLO(pose_model_path)
        print("✅ YOLO-Pose model loaded successfully")
        print(f"   Model type: {type(pose_model.model)}")
        
        # Test basic model info
        print(f"\nBase model classes: {base_model.names}")
        print(f"Pose model classes: {pose_model.names}")
        
        print("\n🎉 Both models loaded successfully! Ready for Streamlit app.")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)