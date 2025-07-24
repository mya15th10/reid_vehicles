"""
Real-time visualization of vehicle re-identification results
"""

import cv2
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import threading
import time
import logging
from kafka import KafkaConsumer
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import queue
import seaborn as sns
from datetime import datetime, timedelta

from config import KAFKA_CONFIG, VIS_CONFIG, CAMERA_CONFIG
from utils import decode_kafka_message, BoundingBox

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleReIDVisualizer:
    """Real-time visualizer for vehicle re-identification results"""
    
    def __init__(self):
        # Data storage
        self.latest_frames = {}  # camera_id -> (frame, timestamp)
        self.reid_results = deque(maxlen=1000)  # Recent ReID results
        self.cross_camera_matches = defaultdict(list)  # Track matches
        self.vehicle_tracks = defaultdict(list)  # vehicle_id -> [positions]
        
        # Statistics
        self.frame_count = defaultdict(int)
        self.vehicle_count = defaultdict(int)
        self.match_count = 0
        
        # Threading
        self.running = False
        self.data_queue = queue.Queue()
        
        # Colors for visualization
        self.colors = VIS_CONFIG['colors']
        self.global_id_colors = {}  # global_id -> color
        self.color_palette = sns.color_palette("husl", 20)
        
        logger.info("VehicleReIDVisualizer initialized")
    
    def _generate_color_for_global_id(self, global_id: str) -> Tuple[int, int, int]:
        """Generate a consistent color for a global vehicle ID"""
        if global_id not in self.global_id_colors:
            # Use hash to get consistent color
            color_idx = hash(global_id) % len(self.color_palette)
            color = self.color_palette[color_idx]
            # Convert to BGR for OpenCV
            self.global_id_colors[global_id] = tuple(int(c * 255) for c in color[::-1])
        
        return self.global_id_colors[global_id]
    
    def _consume_kafka_data(self):
        """Consume data from Kafka topics"""
        # Consumer for video frames
        frame_consumer = KafkaConsumer(
            KAFKA_CONFIG['video_topic'],
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=1000,
            auto_offset_reset='latest'
        )
        
        # Consumer for ReID results
        reid_consumer = KafkaConsumer(
            KAFKA_CONFIG['reid_results_topic'],
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=1000,
            auto_offset_reset='latest'
        )
        
        logger.info("Started Kafka consumers")
        
        while self.running:
            try:
                # Consume frame data
                for message in frame_consumer:
                    if not self.running:
                        break
                    
                    try:
                        frame_data = message.value
                        if frame_data.get('message_type') == 'frame_with_annotations':
                            camera_id = frame_data['camera_id']
                            frame, annotations = decode_kafka_message(frame_data)
                            
                            self.latest_frames[camera_id] = (frame, frame_data['timestamp'])
                            self.frame_count[camera_id] += 1
                            
                            # Add to processing queue
                            self.data_queue.put(('frame', frame_data))
                            
                    except Exception as e:
                        logger.error(f"Error processing frame message: {e}")
                
                # Consume ReID results
                for message in reid_consumer:
                    if not self.running:
                        break
                    
                    try:
                        reid_result = message.value
                        self.reid_results.append(reid_result)
                        
                        # Update statistics
                        camera_id = reid_result['camera_id']
                        self.vehicle_count[camera_id] += 1
                        
                        # Track cross-camera matches
                        global_id = reid_result['global_vehicle_id']
                        if global_id.startswith('global_'):
                            # Check if this global ID exists in other cameras
                            for existing_result in self.reid_results:
                                if (existing_result['global_vehicle_id'] == global_id and 
                                    existing_result['camera_id'] != camera_id):
                                    
                                    match_key = tuple(sorted([camera_id, existing_result['camera_id']]))
                                    if match_key not in [m[:2] for m in self.cross_camera_matches[global_id]]:
                                        self.cross_camera_matches[global_id].append(
                                            (camera_id, existing_result['camera_id'], time.time())
                                        )
                                        self.match_count += 1
                        
                        # Add to processing queue
                        self.data_queue.put(('reid', reid_result))
                        
                    except Exception as e:
                        logger.error(f"Error processing ReID result: {e}")
                
                time.sleep(0.1)  # Brief pause
                
            except Exception as e:
                logger.error(f"Error in Kafka consumer loop: {e}")
                time.sleep(1)
        
        # Close consumers
        frame_consumer.close()
        reid_consumer.close()
        logger.info("Kafka consumers closed")
    
    def _draw_frame_with_reid(self, camera_id: int) -> Optional[np.ndarray]:
        """Draw frame with ReID results"""
        if camera_id not in self.latest_frames:
            return None
        
        frame, timestamp = self.latest_frames[camera_id]
        if frame is None:
            return None
        
        frame_copy = frame.copy()
        
        # Get recent ReID results for this camera
        recent_results = [
            r for r in self.reid_results 
            if r['camera_id'] == camera_id and 
            (time.time() - time.mktime(time.strptime(r['timestamp'][:19], '%Y-%m-%dT%H:%M:%S'))) < 10
        ]
        
        # Draw bounding boxes with global IDs
        for result in recent_results:
            bbox = result['bbox']
            global_id = result['global_vehicle_id']
            vehicle_type = result['vehicle_type']
            
            # Get color for this global ID
            color = self._generate_color_for_global_id(global_id)
            
            # Draw bounding box
            cv2.rectangle(
                frame_copy,
                (int(bbox['xtl']), int(bbox['ytl'])),
                (int(bbox['xbr']), int(bbox['ybr'])),
                color,
                VIS_CONFIG['bbox_thickness']
            )
            
            # Draw label
            label = f"{vehicle_type}: {global_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       VIS_CONFIG['font_scale'], 2)[0]
            
            # Background for text
            cv2.rectangle(
                frame_copy,
                (int(bbox['xtl']), int(bbox['ytl']) - label_size[1] - 10),
                (int(bbox['xtl']) + label_size[0], int(bbox['ytl'])),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                frame_copy, label,
                (int(bbox['xtl']), int(bbox['ytl']) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_CONFIG['font_scale'],
                (255, 255, 255),
                2
            )
        
        # Add camera info
        info_text = f"Camera {camera_id} | Vehicles: {len(recent_results)} | Frame: {self.frame_count[camera_id]}"
        cv2.putText(frame_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def _create_statistics_plot(self) -> np.ndarray:
        """Create a statistics visualization"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Vehicle Re-ID Statistics', fontsize=16)
            
            # Frame counts per camera
            cameras = list(self.frame_count.keys())
            frame_counts = [self.frame_count[c] for c in cameras]
            
            if cameras:
                ax1.bar([f"Camera {c}" for c in cameras], frame_counts)
                ax1.set_title('Frames Processed')
                ax1.set_ylabel('Frame Count')
            
            # Vehicle counts per camera
            vehicle_counts = [self.vehicle_count[c] for c in cameras]
            if cameras:
                ax2.bar([f"Camera {c}" for c in cameras], vehicle_counts)
                ax2.set_title('Vehicles Detected')
                ax2.set_ylabel('Vehicle Count')
            
            # Cross-camera matches
            global_ids = list(self.cross_camera_matches.keys())
            match_counts = [len(self.cross_camera_matches[gid]) for gid in global_ids[:10]]  # Top 10
            
            if global_ids:
                ax3.bar(global_ids[:10], match_counts)
                ax3.set_title('Cross-Camera Matches (Top 10)')
                ax3.set_ylabel('Match Count')
                ax3.tick_params(axis='x', rotation=45)
            
            # ReID results timeline
            if self.reid_results:
                timestamps = []
                camera_ids = []
                
                for result in list(self.reid_results)[-100:]:  # Last 100 results
                    try:
                        ts = datetime.fromisoformat(result['timestamp'][:19])
                        timestamps.append(ts)
                        camera_ids.append(result['camera_id'])
                    except:
                        continue
                
                if timestamps:
                    unique_cameras = list(set(camera_ids))
                    for cam in unique_cameras:
                        cam_timestamps = [ts for ts, cid in zip(timestamps, camera_ids) if cid == cam]
                        if cam_timestamps:
                            ax4.scatter(cam_timestamps, [cam] * len(cam_timestamps), 
                                      label=f'Camera {cam}', alpha=0.7)
                    
                    ax4.set_title('Recent Detection Timeline')
                    ax4.set_ylabel('Camera ID')
                    ax4.legend()
            
            plt.tight_layout()
            
            # Convert to image with proper error handling
            fig.canvas.draw()
            
            # Get canvas dimensions
            width, height = fig.canvas.get_width_height()
            logger.info(f"Statistics plot dimensions: width={width}, height={height}, dpi={fig.dpi}")
            
            # Convert canvas to RGB array
            buf = fig.canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8)
            
            # Reshape with proper dimensions
            try:
                img = img.reshape((height, width, 3))
            except ValueError as e:
                logger.error(f"Error reshaping image: {e}")
                logger.error(f"Buffer size: {len(buf)}, Expected size: {height * width * 3}")
                # Create a fallback image
                img = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(img, 'Statistics Plot Error', (50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            plt.close(fig)
            return img
            
        except Exception as e:
            logger.error(f"Error creating statistics plot: {e}")
            # Return a simple error image
            error_img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(error_img, 'Statistics Plot Error', (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return error_img
    
    def start_visualization(self):
        """Start the visualization system"""
        self.running = True
        
        # Start Kafka consumer thread
        consumer_thread = threading.Thread(target=self._consume_kafka_data)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        logger.info("Visualization started. Press 'q' to quit, 's' for stats.")
        
        try:
            while self.running:
                # Check for available cameras
                available_cameras = list(self.latest_frames.keys())
                
                if not available_cameras:
                    time.sleep(1)
                    continue
                
                # Create display windows for each camera
                display_frames = []
                
                for camera_id in sorted(available_cameras):
                    frame_with_reid = self._draw_frame_with_reid(camera_id)
                    if frame_with_reid is not None:
                        # Resize for display
                        h, w = frame_with_reid.shape[:2]
                        if w > VIS_CONFIG['display_width']:
                            scale = VIS_CONFIG['display_width'] / w
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            frame_with_reid = cv2.resize(frame_with_reid, (new_w, new_h))
                        
                        display_frames.append(frame_with_reid)
                
                # Display frames
                if len(display_frames) == 1:
                    cv2.imshow('Vehicle ReID - Camera View', display_frames[0])
                elif len(display_frames) == 2:
                    # Side by side
                    combined = np.hstack(display_frames)
                    cv2.imshow('Vehicle ReID - Multi Camera', combined)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Show statistics
                    stats_img = self._create_statistics_plot()
                    cv2.imshow('Statistics', cv2.cvtColor(stats_img, cv2.COLOR_RGB2BGR))
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the visualization"""
        logger.info("Stopping visualization...")
        self.running = False
        cv2.destroyAllWindows()
        logger.info("Visualization stopped")
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return {
            'frames_processed': dict(self.frame_count),
            'vehicles_detected': dict(self.vehicle_count),
            'cross_camera_matches': self.match_count,
            'total_reid_results': len(self.reid_results),
            'active_global_ids': len(self.cross_camera_matches),
            'latest_frames_available': list(self.latest_frames.keys())
        }

def main():
    """Main function to run the visualizer"""
    logger.info("Starting Vehicle ReID Visualizer")
    
    visualizer = VehicleReIDVisualizer()
    
    try:
        visualizer.start_visualization()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        visualizer.stop()
        logger.info("Visualizer shutdown complete")

if __name__ == "__main__":
    main()
