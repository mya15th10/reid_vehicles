"""
Utility functions for parsing XML annotations and processing video data
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BoundingBox:
    """Bounding box with metadata"""
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    label: str
    vehicle_id: str
    frame_id: int
    camera_id: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'xtl': self.xtl,
            'ytl': self.ytl,
            'xbr': self.xbr,
            'ybr': self.ybr,
            'label': self.label,
            'vehicle_id': self.vehicle_id,
            'frame_id': self.frame_id,
            'camera_id': self.camera_id,
            'confidence': self.confidence
        }
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xtl + self.xbr) / 2, (self.ytl + self.ybr) / 2)
    
    @property
    def width(self) -> float:
        return abs(self.xbr - self.xtl)
    
    @property
    def height(self) -> float:
        return abs(self.ybr - self.ytl)
    
    @property
    def area(self) -> float:
        return self.width * self.height

class AnnotationParser:
    """Parser for CVAT XML annotation files"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.camera_id = self._extract_camera_id()
        
    def _extract_camera_id(self) -> int:
        """Extract camera ID from the annotation file"""
        task_name_elem = self.root.find('.//task/n')
        if task_name_elem is not None and task_name_elem.text:
            task_name = task_name_elem.text
            if 'Camera11' in task_name:
                return 11
            elif 'Camera21' in task_name:
                return 21
            else:
                # Try to extract number from task name
                import re
                match = re.search(r'(\d+)', task_name)
                return int(match.group(1)) if match else 0
        else:
            # Extract from file path as fallback
            if 'annotations_11' in self.xml_path:
                return 11
            elif 'annotations_21' in self.xml_path:
                return 21
            else:
                return 0
    
    def get_frame_annotations(self, frame_id: int) -> List[BoundingBox]:
        """Get all vehicle annotations for a specific frame"""
        annotations = []
        
        # Find the image element for this frame
        image_elem = self.root.find(f".//image[@id='{frame_id}']")
        if image_elem is None:
            return annotations
        
        # Extract all bounding boxes from this frame
        for box in image_elem.findall('box'):
            label = box.get('label')
            if label in ['car', 'truck', 'bus', 'bicycle']:  # Vehicle types
                # Get bounding box coordinates
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                # Get vehicle ID from attributes
                vehicle_id = None
                id_attr = box.find("attribute[@name='id']")
                if id_attr is not None and id_attr.text and id_attr.text.strip():
                    vehicle_id = id_attr.text.strip()
                else:
                    # Generate a unique ID based on frame and box index if missing
                    box_index = list(image_elem.findall('box')).index(box)
                    vehicle_id = f"auto_{frame_id}_{box_index}"
                
                if vehicle_id:
                    bbox = BoundingBox(
                        xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr,
                        label=label,
                        vehicle_id=f"{self.camera_id}_{vehicle_id}",
                        frame_id=frame_id,
                        camera_id=self.camera_id
                    )
                    annotations.append(bbox)
        
        return annotations
    
    def get_all_annotations(self) -> Dict[int, List[BoundingBox]]:
        """Get all annotations organized by frame ID"""
        all_annotations = {}
        
        for image in self.root.findall('.//image'):
            frame_id = int(image.get('id'))
            annotations = self.get_frame_annotations(frame_id)
            if annotations:
                all_annotations[frame_id] = annotations
                
        return all_annotations
    
    def get_vehicle_tracks(self) -> Dict[str, List[BoundingBox]]:
        """Get vehicle tracks organized by vehicle ID"""
        tracks = {}
        all_annotations = self.get_all_annotations()
        
        for frame_id, frame_annotations in all_annotations.items():
            for bbox in frame_annotations:
                if bbox.vehicle_id not in tracks:
                    tracks[bbox.vehicle_id] = []
                tracks[bbox.vehicle_id].append(bbox)
        
        # Sort tracks by frame ID
        for vehicle_id in tracks:
            tracks[vehicle_id].sort(key=lambda x: x.frame_id)
            
        return tracks

class VideoProcessor:
    """Utility class for video processing operations"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get a specific frame by ID"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def extract_vehicle_crop(self, frame: np.ndarray, bbox: BoundingBox, 
                           padding: int = 10) -> Optional[np.ndarray]:
        """Extract vehicle crop from frame using bounding box"""
        h, w = frame.shape[:2]
        
        # Add padding and ensure bounds
        x1 = max(0, int(bbox.xtl - padding))
        y1 = max(0, int(bbox.ytl - padding))
        x2 = min(w, int(bbox.xbr + padding))
        y2 = min(h, int(bbox.ybr + padding))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2]
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame maintaining aspect ratio"""
        return cv2.resize(frame, target_size)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

def create_kafka_message(frame_data: np.ndarray, annotations: List[BoundingBox], 
                        camera_id: int, frame_id: int, timestamp: str = None) -> Dict:
    """Create a standardized Kafka message"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Encode frame as JPEG for transmission
    _, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_bytes = buffer.tobytes()
    
    message = {
        'timestamp': timestamp,
        'camera_id': camera_id,
        'frame_id': frame_id,
        'frame_data': frame_bytes.hex(),  # Hex encode for JSON serialization
        'annotations': [ann.to_dict() for ann in annotations],
        'frame_shape': frame_data.shape,
        'message_type': 'frame_with_annotations'
    }
    
    return message

def decode_kafka_message(message_dict: Dict) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Decode a Kafka message back to frame and annotations"""
    # Decode frame
    frame_bytes = bytes.fromhex(message_dict['frame_data'])
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    
    # Decode annotations
    annotations = []
    for ann_dict in message_dict['annotations']:
        bbox = BoundingBox(**ann_dict)
        annotations.append(bbox)
    
    return frame, annotations

def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    # Calculate intersection
    x1 = max(bbox1.xtl, bbox2.xtl)
    y1 = max(bbox1.ytl, bbox2.ytl)
    x2 = min(bbox1.xbr, bbox2.xbr)
    y2 = min(bbox1.ybr, bbox2.ybr)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    union = bbox1.area + bbox2.area - intersection
    
    return intersection / union if union > 0 else 0.0

def draw_bbox_on_frame(frame: np.ndarray, bbox: BoundingBox, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      thickness: int = 2) -> np.ndarray:
    """Draw bounding box on frame with label"""
    frame_copy = frame.copy()
    
    # Draw rectangle
    cv2.rectangle(frame_copy, 
                 (int(bbox.xtl), int(bbox.ytl)), 
                 (int(bbox.xbr), int(bbox.ybr)), 
                 color, thickness)
    
    # Draw label
    label_text = f"{bbox.label}:{bbox.vehicle_id}"
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Background for text
    cv2.rectangle(frame_copy,
                 (int(bbox.xtl), int(bbox.ytl) - label_size[1] - 10),
                 (int(bbox.xtl) + label_size[0], int(bbox.ytl)),
                 color, -1)
    
    # Text
    cv2.putText(frame_copy, label_text,
               (int(bbox.xtl), int(bbox.ytl) - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_copy

def resize_frame_to_annotation_resolution(frame: np.ndarray, target_resolution: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
    """
    Resize frame to match annotation resolution.
    Default resolution is 1080x1920 based on the XML annotation metadata.
    """
    if frame is None:
        return None
    
    current_height, current_width = frame.shape[:2]
    target_width, target_height = target_resolution
    
    # Only resize if different from target resolution
    if current_width != target_width or current_height != target_height:
        frame = cv2.resize(frame, (target_width, target_height))
    
    return frame
