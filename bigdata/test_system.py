"""
Test script to verify the Vehicle ReID system components
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CAMERA_CONFIG, DATA_PATHS
from utils import AnnotationParser, VideoProcessor, BoundingBox, create_kafka_message, decode_kafka_message
from reid_model import VehicleReIDModel, CrossCameraReID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_annotation_parsing():
    """Test XML annotation parsing"""
    print("Testing annotation parsing...")
    
    for camera_key, camera_config in CAMERA_CONFIG.items():
        try:
            parser = AnnotationParser(camera_config['annotations_path'])
            print(f"\n{camera_config['name']} (Camera {camera_config['id']}):")
            
            # Test getting all annotations
            all_annotations = parser.get_all_annotations()
            print(f"  Total frames with annotations: {len(all_annotations)}")
            
            # Test getting vehicle tracks
            tracks = parser.get_vehicle_tracks()
            print(f"  Total vehicle tracks: {len(tracks)}")
            
            # Show sample annotations
            if all_annotations:
                sample_frame = list(all_annotations.keys())[0]
                sample_annotations = all_annotations[sample_frame]
                print(f"  Sample frame {sample_frame}: {len(sample_annotations)} vehicles")
                
                for ann in sample_annotations[:3]:  # Show first 3
                    print(f"    {ann.label} ID:{ann.vehicle_id} "
                          f"bbox:({ann.xtl:.1f},{ann.ytl:.1f},{ann.xbr:.1f},{ann.ybr:.1f})")
            
        except Exception as e:
            print(f"Error testing {camera_key}: {e}")
    
    print("✓ Annotation parsing test completed")

def test_video_processing():
    """Test video processing"""
    print("\nTesting video processing...")
    
    for camera_key, camera_config in CAMERA_CONFIG.items():
        try:
            processor = VideoProcessor(camera_config['video_path'])
            print(f"\n{camera_config['name']}:")
            print(f"  Frame count: {processor.frame_count}")
            print(f"  FPS: {processor.fps:.2f}")
            print(f"  Resolution: {processor.width}x{processor.height}")
            
            # Test frame extraction
            frame = processor.get_frame(0)
            if frame is not None:
                print(f"  Successfully extracted frame 0: shape {frame.shape}")
            else:
                print("  Failed to extract frame 0")
            
        except Exception as e:
            print(f"Error testing video for {camera_key}: {e}")
    
    print("✓ Video processing test completed")

def test_reid_model():
    """Test ReID model"""
    print("\nTesting ReID model...")
    
    try:
        # Initialize model
        reid_model = VehicleReIDModel()
        print(f"Model initialized on device: {reid_model.device}")
        
        # Create dummy vehicle crop
        dummy_crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        
        # Extract features
        features = reid_model.extract_features(dummy_crop)
        print(f"Feature extraction successful: shape {features.shape}")
        
        # Test gallery operations
        reid_model.add_to_gallery("test_vehicle_1", features, {"camera_id": 11, "vehicle_type": "car"})
        
        # Test matching
        matches = reid_model.find_matches(features, threshold=0.5)
        print(f"Found {len(matches)} matches (expected 1)")
        
        # Test cross-camera ReID
        cross_camera_reid = CrossCameraReID(reid_model)
        global_id = cross_camera_reid.process_vehicle(
            dummy_crop, "test_vehicle_2", 21, "car", 0
        )
        print(f"Cross-camera processing successful: global_id = {global_id}")
        
        print("✓ ReID model test completed")
        
    except Exception as e:
        print(f"Error testing ReID model: {e}")

def test_kafka_message_serialization():
    """Test Kafka message creation and parsing"""
    print("\nTesting Kafka message serialization...")
    
    try:
        # Create test frame and annotations
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_annotations = [
            BoundingBox(10, 10, 50, 50, "car", "test_vehicle_1", 0, 11),
            BoundingBox(60, 60, 90, 90, "truck", "test_vehicle_2", 0, 11)
        ]
        
        # Create message
        message = create_kafka_message(test_frame, test_annotations, 11, 0)
        print(f"Message created successfully")
        print(f"  Frame data size: {len(message['frame_data'])} bytes (hex)")
        print(f"  Annotations: {len(message['annotations'])}")
        
        # Test decoding
        decoded_frame, decoded_annotations = decode_kafka_message(message)
        print(f"Message decoded successfully")
        print(f"  Decoded frame shape: {decoded_frame.shape}")
        print(f"  Decoded annotations: {len(decoded_annotations)}")
        
        # Verify data integrity
        if np.array_equal(test_frame, decoded_frame):
            print("✓ Frame data integrity verified")
        else:
            print("✗ Frame data integrity check failed")
        
        if len(test_annotations) == len(decoded_annotations):
            print("✓ Annotation count verified")
        else:
            print("✗ Annotation count mismatch")
        
        print("✓ Kafka message serialization test completed")
        
    except Exception as e:
        print(f"Error testing Kafka message serialization: {e}")

def test_end_to_end_sample():
    """Test end-to-end processing with real data"""
    print("\nTesting end-to-end sample processing...")
    
    try:
        # Use Camera11 as test
        camera_config = CAMERA_CONFIG['camera1']
        
        # Initialize components
        parser = AnnotationParser(camera_config['annotations_path'])
        processor = VideoProcessor(camera_config['video_path'])
        reid_model = VehicleReIDModel()
        cross_camera_reid = CrossCameraReID(reid_model)
        
        # Get first frame with annotations
        all_annotations = parser.get_all_annotations()
        if not all_annotations:
            print("No annotations found for testing")
            return
        
        test_frame_id = list(all_annotations.keys())[0]
        annotations = all_annotations[test_frame_id]
        
        print(f"Testing with frame {test_frame_id}, {len(annotations)} vehicles")
        
        # Get frame
        frame = processor.get_frame(test_frame_id)
        if frame is None:
            print("Could not extract test frame")
            return
        
        print(f"Frame extracted: {frame.shape}")
        
        # Process each vehicle
        results = []
        for annotation in annotations:
            # Extract vehicle crop
            x1, y1 = int(annotation.xtl), int(annotation.ytl)
            x2, y2 = int(annotation.xbr), int(annotation.ybr)
            
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
                vehicle_crop = frame[y1:y2, x1:x2]
                
                # Process with ReID
                global_id = cross_camera_reid.process_vehicle(
                    vehicle_crop, annotation.vehicle_id, annotation.camera_id, 
                    annotation.label, test_frame_id
                )
                
                result = {
                    'local_id': annotation.vehicle_id,
                    'global_id': global_id,
                    'type': annotation.label,
                    'bbox': (x1, y1, x2, y2)
                }
                results.append(result)
                
                print(f"  Processed {annotation.label} {annotation.vehicle_id} -> {global_id}")
        
        print(f"✓ End-to-end test completed: {len(results)} vehicles processed")
        
        # Test cross-camera matching with second camera
        if len(CAMERA_CONFIG) > 1:
            camera2_config = CAMERA_CONFIG['camera2']
            parser2 = AnnotationParser(camera2_config['annotations_path'])
            processor2 = VideoProcessor(camera2_config['video_path'])
            
            all_annotations2 = parser2.get_all_annotations()
            if all_annotations2:
                test_frame_id2 = list(all_annotations2.keys())[0]
                annotations2 = all_annotations2[test_frame_id2]
                frame2 = processor2.get_frame(test_frame_id2)
                
                if frame2 is not None and annotations2:
                    print(f"\nTesting cross-camera with Camera21 frame {test_frame_id2}")
                    
                    for annotation in annotations2[:2]:  # Test first 2 vehicles
                        x1, y1 = int(annotation.xtl), int(annotation.ytl)
                        x2, y2 = int(annotation.xbr), int(annotation.ybr)
                        
                        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 < frame2.shape[1] and y2 < frame2.shape[0]:
                            vehicle_crop = frame2[y1:y2, x1:x2]
                            
                            global_id = cross_camera_reid.process_vehicle(
                                vehicle_crop, annotation.vehicle_id, annotation.camera_id,
                                annotation.label, test_frame_id2
                            )
                            
                            print(f"  Camera21 {annotation.label} {annotation.vehicle_id} -> {global_id}")
                    
                    # Check for cross-camera matches
                    matches = cross_camera_reid.get_cross_camera_matches()
                    print(f"Cross-camera matches found: {len(matches)}")
        
        print("✓ Cross-camera test completed")
        
    except Exception as e:
        print(f"Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()

def test_system_dependencies():
    """Test system dependencies"""
    print("Testing system dependencies...")
    
    # Test imports
    dependencies = [
        ('cv2', 'OpenCV'),
        ('torch', 'PyTorch'),
        ('kafka', 'Kafka Python'),
        ('pyspark', 'PySpark'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name} available")
        except ImportError:
            print(f"  ✗ {name} not available")
    
    # Test file paths
    print("\nTesting file paths...")
    for camera_key, camera_config in CAMERA_CONFIG.items():
        video_path = camera_config['video_path']
        annotations_path = camera_config['annotations_path']
        
        if os.path.exists(video_path):
            print(f"  ✓ {camera_config['name']} video found")
        else:
            print(f"  ✗ {camera_config['name']} video not found: {video_path}")
        
        if os.path.exists(annotations_path):
            print(f"  ✓ {camera_config['name']} annotations found")
        else:
            print(f"  ✗ {camera_config['name']} annotations not found: {annotations_path}")

def main():
    """Run all tests"""
    print("Vehicle ReID System Test Suite")
    print("=" * 40)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Run tests
    test_system_dependencies()
    test_annotation_parsing()
    test_video_processing()
    test_reid_model()
    test_kafka_message_serialization()
    test_end_to_end_sample()
    
    print("\n" + "=" * 40)
    print("Test suite completed!")
    print("If all tests passed, the system is ready to run.")
    print("\nTo start the system:")
    print("1. Run: ./setup_environment.sh")
    print("2. In separate terminals:")
    print("   - python producer.py")
    print("   - python consumer.py")
    print("   - python visualizer.py")

if __name__ == "__main__":
    main()
