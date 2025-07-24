"""
Kafka Producer for streaming video frames and vehicle detection data
"""

import cv2
import json
import time
import logging
from kafka import KafkaProducer
from typing import Dict, List
import threading
from datetime import datetime
import signal
import sys

from config import KAFKA_CONFIG, CAMERA_CONFIG, PROCESSING_CONFIG, DATA_PATHS
from utils import AnnotationParser, VideoProcessor, create_kafka_message, BoundingBox, resize_frame_to_annotation_resolution

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleKafkaProducer:
    """Kafka producer for streaming vehicle data from multiple cameras"""
    
    def __init__(self):
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            max_request_size=10485760,  # 10MB for large frames
            compression_type='gzip'
        )
        
        # Initialize video processors and annotation parsers
        self.video_processors = {}
        self.annotation_parsers = {}
        
        for camera_key, camera_config in CAMERA_CONFIG.items():
            try:
                self.video_processors[camera_key] = VideoProcessor(camera_config['video_path'])
                self.annotation_parsers[camera_key] = AnnotationParser(camera_config['annotations_path'])
                logger.info(f"Initialized {camera_key}: {camera_config['name']}")
            except Exception as e:
                logger.error(f"Failed to initialize {camera_key}: {e}")
        
        # Control variables
        self.running = False
        self.threads = []
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("VehicleKafkaProducer initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping producer...")
        self.stop()
        sys.exit(0)
    def _produce_camera_stream(self, camera_key: str):
        """Produce stream for a single camera"""
        camera_config = CAMERA_CONFIG[camera_key]
        video_processor = self.video_processors[camera_key]
        annotation_parser = self.annotation_parsers[camera_key]

        camera_id = camera_config['id']
        camera_name = camera_config['name']

        logger.info(f"Starting stream for {camera_name} (ID: {camera_id})")

        all_annotations = annotation_parser.get_all_annotations()
        annotated_frame_ids = sorted(all_annotations.keys())

        mode = PROCESSING_CONFIG.get('mode', 'real_time')  # 'real_time' or 'video_second'
        fps_target = PROCESSING_CONFIG['fps']
        frame_interval = 1.0 / fps_target
        video_fps = int(video_processor.fps)

        logger.info(f"{camera_name}: Running in '{mode}' mode")

        for frame_id in annotated_frame_ids:
            if not self.running:
                break

            try:
                start_time = time.time()

                frame_index = frame_id * video_fps  # actual frame in video file
                frame = video_processor.get_frame(frame_index)
                if frame is None:
                    logger.warning(f"Missing video frame {frame_index} for annotation frame {frame_id}")
                    continue

                # Resize frame to 1080x1920 for consistency
                frame = resize_frame_to_annotation_resolution(frame)

                # Get annotations
                annotations = all_annotations.get(frame_id, [])
                if annotations:
                    target_size = PROCESSING_CONFIG['frame_resize']
                    if frame.shape[:2] != target_size[::-1]:
                        frame_resized = cv2.resize(frame, target_size)
                        scale_x = target_size[0] / frame.shape[1]
                        scale_y = target_size[1] / frame.shape[0]

                        scaled_annotations = []
                        for ann in annotations:
                            scaled_ann = BoundingBox(
                                xtl=ann.xtl * scale_x,
                                ytl=ann.ytl * scale_y,
                                xbr=ann.xbr * scale_x,
                                ybr=ann.ybr * scale_y,
                                label=ann.label,
                                vehicle_id=ann.vehicle_id,
                                frame_id=ann.frame_id,
                                camera_id=ann.camera_id,
                                confidence=ann.confidence
                            )
                            scaled_annotations.append(scaled_ann)
                        annotations = scaled_annotations
                    else:
                        frame_resized = frame

                    # Send to Kafka
                    timestamp = datetime.now().isoformat()
                    message = create_kafka_message(
                        frame_data=frame_resized,
                        annotations=annotations,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        timestamp=timestamp
                    )

                    self.producer.send(
                        KAFKA_CONFIG['video_topic'],
                        key=f"{camera_id}_{frame_id}",
                        value=message
                    )

                    detection_message = {
                        'timestamp': timestamp,
                        'camera_id': camera_id,
                        'frame_id': frame_id,
                        'detections': [ann.to_dict() for ann in annotations],
                        'message_type': 'detections_only'
                    }

                    self.producer.send(
                        KAFKA_CONFIG['detection_topic'],
                        key=f"{camera_id}_{frame_id}",
                        value=detection_message
                    )

                    logger.info(f"{camera_name}: Frame {frame_id} (video frame {frame_index}) sent with {len(annotations)} vehicles")

                # Optional real-time sleep
                if mode == 'real_time':
                    elapsed = time.time() - start_time
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error processing annotation frame {frame_id} (video frame {frame_index}) for {camera_name}: {e}")
                time.sleep(0.1)

        logger.info(f"Stream ended for {camera_name}")

    # def _produce_camera_stream(self, camera_key: str):
        # """Produce stream for a single camera"""
        # camera_config = CAMERA_CONFIG[camera_key]
        # video_processor = self.video_processors[camera_key]
        # annotation_parser = self.annotation_parsers[camera_key]
        
        # camera_id = camera_config['id']
        # camera_name = camera_config['name']
        
        # logger.info(f"Starting stream for {camera_name} (ID: {camera_id})")
        
        # # Get all annotations for efficient lookup
        # all_annotations = annotation_parser.get_all_annotations()
        # max_frame = max(all_annotations.keys()) if all_annotations else video_processor.frame_count - 1
        
        # frame_id = 0
        # fps_target = PROCESSING_CONFIG['fps']
        # frame_interval = 1.0 / fps_target
        
        # while self.running and frame_id <= max_frame:
        #     try:
        #         start_time = time.time()
                
        #         # Get frame from video
        #         frame = video_processor.get_frame(frame_id)
        #         if frame is None:
        #             logger.warning(f"Could not read frame {frame_id} from {camera_name}")
        #             frame_id += 1
        #             continue
                
        #         # Get annotations for this frame
        #         annotations = all_annotations.get(frame_id, [])
                
        #         if annotations:  # Only send frames with vehicle detections
        #             # Resize frame for efficiency
        #             target_size = PROCESSING_CONFIG['frame_resize']
        #             if frame.shape[:2] != target_size[::-1]:  # height, width vs width, height
        #                 frame_resized = cv2.resize(frame, target_size)
                        
        #                 # Scale annotations accordingly
        #                 scale_x = target_size[0] / frame.shape[1]
        #                 scale_y = target_size[1] / frame.shape[0]
                        
        #                 scaled_annotations = []
        #                 for ann in annotations:
        #                     scaled_ann = BoundingBox(
        #                         xtl=ann.xtl * scale_x,
        #                         ytl=ann.ytl * scale_y,
        #                         xbr=ann.xbr * scale_x,
        #                         ybr=ann.ybr * scale_y,
        #                         label=ann.label,
        #                         vehicle_id=ann.vehicle_id,
        #                         frame_id=ann.frame_id,
        #                         camera_id=ann.camera_id,
        #                         confidence=ann.confidence
        #                     )
        #                     scaled_annotations.append(scaled_ann)
        #                 annotations = scaled_annotations
        #             else:
        #                 frame_resized = frame
                    
        #             # Create Kafka message
        #             timestamp = datetime.now().isoformat()
        #             message = create_kafka_message(
        #                 frame_data=frame_resized,
        #                 annotations=annotations,
        #                 camera_id=camera_id,
        #                 frame_id=frame_id,
        #                 timestamp=timestamp
        #             )
                    
        #             # Send to Kafka
        #             try:
        #                 self.producer.send(
        #                     KAFKA_CONFIG['video_topic'],
        #                     key=f"{camera_id}_{frame_id}",
        #                     value=message
        #                 )
                        
        #                 # Also send detections to separate topic
        #                 detection_message = {
        #                     'timestamp': timestamp,
        #                     'camera_id': camera_id,
        #                     'frame_id': frame_id,
        #                     'detections': [ann.to_dict() for ann in annotations],
        #                     'message_type': 'detections_only'
        #                 }
                        
        #                 self.producer.send(
        #                     KAFKA_CONFIG['detection_topic'],
        #                     key=f"{camera_id}_{frame_id}",
        #                     value=detection_message
        #                 )
                        
        #                 logger.info(f"{camera_name}: Frame {frame_id} sent with {len(annotations)} vehicles")
                        
        #             except Exception as e:
        #                 logger.error(f"Error sending message for {camera_name} frame {frame_id}: {e}")
                
        #         frame_id += 1
                
        #         # Control frame rate
        #         elapsed = time.time() - start_time
        #         sleep_time = max(0, frame_interval - elapsed)
        #         if sleep_time > 0:
        #             time.sleep(sleep_time)
                
        #     except Exception as e:
        #         logger.error(f"Error processing frame {frame_id} for {camera_name}: {e}")
        #         frame_id += 1
        #         time.sleep(0.1)  # Brief pause before continuing
        
        # logger.info(f"Stream ended for {camera_name}")
    
    def start(self):
        """Start producing streams for all cameras"""
        if self.running:
            logger.warning("Producer is already running")
            return
        
        self.running = True
        
        # Start a thread for each camera
        for camera_key in self.video_processors.keys():
            thread = threading.Thread(
                target=self._produce_camera_stream,
                args=(camera_key,),
                name=f"Producer-{camera_key}"
            )
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            logger.info(f"Started producer thread for {camera_key}")
        
        logger.info(f"All producer threads started ({len(self.threads)} cameras)")
    
    def stop(self):
        """Stop all producer threads"""
        logger.info("Stopping producer...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not stop gracefully")
        
        # Close Kafka producer
        self.producer.flush()
        self.producer.close()
        
        logger.info("Producer stopped")
    
    def get_stats(self) -> Dict:
        """Get producer statistics"""
        stats = {
            'running': self.running,
            'active_threads': len([t for t in self.threads if t.is_alive()]),
            'total_threads': len(self.threads),
            'cameras': {}
        }
        
        for camera_key, video_processor in self.video_processors.items():
            camera_config = CAMERA_CONFIG[camera_key]
            stats['cameras'][camera_key] = {
                'name': camera_config['name'],
                'id': camera_config['id'],
                'frame_count': video_processor.frame_count,
                'fps': video_processor.fps,
                'resolution': (video_processor.width, video_processor.height)
            }
        
        return stats

def main():
    """Main function to run the producer"""
    logger.info("Starting Vehicle Kafka Producer")
    
    # Create and start producer
    producer = VehicleKafkaProducer()
    
    try:
        # Display stats
        stats = producer.get_stats()
        logger.info("Producer Statistics:")
        logger.info(f"  Cameras: {len(stats['cameras'])}")
        for camera_key, camera_info in stats['cameras'].items():
            logger.info(f"    {camera_info['name']}: {camera_info['frame_count']} frames, "
                       f"{camera_info['fps']:.1f} FPS, {camera_info['resolution']}")
        
        # Start streaming
        producer.start()
        
        # Keep main thread alive
        logger.info("Producer is running. Press Ctrl+C to stop.")
        while producer.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        producer.stop()
        logger.info("Producer shutdown complete")

if __name__ == "__main__":
    main()
