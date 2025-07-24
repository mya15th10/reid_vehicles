"""
Configuration settings for the Vehicle ReID system
"""

import os

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': ['localhost:9092'],
    'video_topic': 'vehicle_video_frames',
    'detection_topic': 'vehicle_detections',
    'reid_results_topic': 'reid_results',
    'consumer_group': 'vehicle_reid_group'
}

# Spark Configuration
SPARK_CONFIG = {
    'app_name': 'VehicleReIDSystem',
    'master': 'local[*]',
    'packages': [
        'org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0',
        'org.apache.kafka:kafka-clients:2.8.0'
    ],
    'checkpoint_location': './checkpoint'
}

# Video and Data Paths
DATA_PATHS = {
    'video1': '../reid/video1(1).MOV',
    'video2': '../reid/video2(1).MOV',
    'annotations1': '../reid/annotations_11.xml',
    'annotations2': '../reid/annotations_21.xml',
    'output_dir': './output'
}

# Camera Configuration
CAMERA_CONFIG = {
    'camera1': {
        'id': 11,
        'name': 'Camera11',
        'video_path': DATA_PATHS['video1'],
        'annotations_path': DATA_PATHS['annotations1']
    },
    'camera2': {
        'id': 21,
        'name': 'Camera21',
        'video_path': DATA_PATHS['video2'],
        'annotations_path': DATA_PATHS['annotations2']
    }
}

# ReID Model Configuration
REID_CONFIG = {
    'model_name': 'resnet50',
    'pretrained': True,
    'feature_dim': 2048,
    'similarity_threshold': 0.7,
    'max_distance': 0.5
}

# Processing Configuration
PROCESSING_CONFIG = {
    'batch_size': 32,
    'fps': 10,  # Process every 10th frame for efficiency
    'frame_resize': (224, 224),
    'max_vehicles_per_frame': 50,
    'mode': 'video_second',  # 'real_time' or 'video_second'
    'max_offsets_per_trigger': 1000,  # For Kafka streaming
    'enable_pandas_udf': True,  # Enable Pandas UDF for better performance
    'pandas_udf_batch_size': 100  # Batch size for Pandas UDF processing
}

# Visualization Configuration
VIS_CONFIG = {
    'display_width': 1080,
    'display_height': 720,
    'bbox_thickness': 2,
    'font_scale': 0.7,
    'colors': {
        'car': (0, 255, 0),      # Green
        'truck': (255, 0, 0),    # Red
        'bus': (0, 0, 255),      # Blue
        'bicycle': (255, 255, 0)  # Yellow
    }
}

# Create output directory if it doesn't exist
os.makedirs(DATA_PATHS['output_dir'], exist_ok=True)
os.makedirs(SPARK_CONFIG['checkpoint_location'], exist_ok=True)
