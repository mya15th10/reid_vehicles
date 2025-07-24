"""
Spark-based Kafka Consumer for Vehicle Re-Identification
"""

import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import collect_list, explode, udf
from pyspark.sql.types import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
try:
    from pyspark.sql.functions import pandas_udf, col, window
    from pyspark.sql.pandas.functions import PandasUDFType
    PANDAS_UDF_AVAILABLE = True
except ImportError:
    PANDAS_UDF_AVAILABLE = False
    PandasUDFType = None
    print("Pandas UDF not available, falling back to regular processing")

import pandas as pd
import numpy as np
import cv2
import base64
import os
import time
import traceback
import builtins
from typing import Dict, List, Any, Iterator
import pickle

from config import KAFKA_CONFIG, SPARK_CONFIG, REID_CONFIG, PROCESSING_CONFIG
from utils import decode_kafka_message, BoundingBox
from reid_model import VehicleReIDModel, CrossCameraReID

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleReIDSparkConsumer:
    """Spark-based consumer for vehicle re-identification"""
    
    def __init__(self):
        # Initialize Spark session
        self.spark = self._create_spark_session()
        
        # Initialize ReID system
        self.reid_model = VehicleReIDModel(
            model_name=REID_CONFIG['model_name'],
            pretrained=REID_CONFIG['pretrained'],
            feature_dim=REID_CONFIG['feature_dim']
        )
        self.cross_camera_reid = CrossCameraReID(
            reid_model=self.reid_model,
            similarity_threshold=REID_CONFIG['similarity_threshold']
        )
        
        # Try to load existing state
        state_dir = './reid_state'
        if os.path.exists(state_dir):
            self.cross_camera_reid.load_state(state_dir)
            logger.info("Loaded existing ReID state")
        
        # Results storage
        self.processed_frames = 0
        self.total_vehicles = 0
        self.cross_camera_matches_count = 0
        
        logger.info("VehicleReIDSparkConsumer initialized")
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        builder = SparkSession.builder \
            .appName(SPARK_CONFIG['app_name']) \
            .master(SPARK_CONFIG['master'])
        
        # Add Kafka packages
        for package in SPARK_CONFIG['packages']:
            builder = builder.config("spark.jars.packages", package)
        
        # Additional Spark configurations
        spark = builder \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session created")
        return spark
    
    def _process_vehicle_batch(self, df: DataFrame, epoch_id: int):
        """Process a batch of vehicle data for ReID"""
        try:
            logger.info(f"Processing batch {epoch_id}")
            
            # Collect the batch data
            batch_data = df.collect()
            
            if not batch_data:
                logger.info(f"Batch {epoch_id}: No data to process")
                return
            
            batch_results = []
            
            for row in batch_data:
                try:
                    # Parse the message
                    message_dict = json.loads(row.value)
                    
                    if message_dict.get('message_type') != 'frame_with_annotations':
                        continue
                    
                    # Decode frame and annotations
                    frame, annotations = decode_kafka_message(message_dict)
                    
                    camera_id = message_dict['camera_id']
                    frame_id = message_dict['frame_id']
                    timestamp = message_dict['timestamp']
                    
                    logger.info(f"Processing frame {frame_id} from camera {camera_id} with {len(annotations)} vehicles")
                    
                    frame_results = []
                    
                    # Process each vehicle in the frame
                    for annotation in annotations:
                        try:
                            # Extract vehicle crop
                            x1, y1 = int(annotation.xtl), int(annotation.ytl)
                            x2, y2 = int(annotation.xbr), int(annotation.ybr)
                            
                            if x2 > x1 and y2 > y1:
                                vehicle_crop = frame[y1:y2, x1:x2]
                                
                                # Perform ReID
                                global_id = self.cross_camera_reid.process_vehicle(
                                    vehicle_crop=vehicle_crop,
                                    vehicle_id=annotation.vehicle_id,
                                    camera_id=camera_id,
                                    vehicle_type=annotation.label,
                                    frame_id=frame_id
                                )
                                
                                # Create result
                                result = {
                                    'timestamp': timestamp,
                                    'camera_id': camera_id,
                                    'frame_id': frame_id,
                                    'local_vehicle_id': annotation.vehicle_id,
                                    'global_vehicle_id': global_id,
                                    'vehicle_type': annotation.label,
                                    'bbox': {
                                        'xtl': annotation.xtl,
                                        'ytl': annotation.ytl,
                                        'xbr': annotation.xbr,
                                        'ybr': annotation.ybr
                                    },
                                    'confidence': annotation.confidence
                                }
                                
                                frame_results.append(result)
                                self.total_vehicles += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing vehicle {annotation.vehicle_id}: {e}")
                    
                    if frame_results:
                        batch_results.extend(frame_results)
                    
                    self.processed_frames += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row in batch {epoch_id}: {e}")
            
            # Send results to Kafka
            if batch_results:
                self._send_results_to_kafka(batch_results)
            
            # Save state periodically
            if self.processed_frames % 50 == 0:
                self._save_state()
            
            # Update match count
            matches = self.cross_camera_reid.get_cross_camera_matches()
            total_matches = sum(len(match_list) for match_list in matches.values())
            self.cross_camera_matches_count = total_matches
            
            logger.info(f"Batch {epoch_id} completed: {len(batch_results)} vehicles processed, "
                       f"{self.cross_camera_matches_count} total cross-camera matches")
            
        except Exception as e:
            logger.error(f"Error in batch processing {epoch_id}: {e}")
    
    def _send_results_to_kafka(self, results: List[Dict]):
        """Send ReID results back to Kafka"""
        try:
            from kafka import KafkaProducer
            
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            for result in results:
                producer.send(
                    KAFKA_CONFIG['reid_results_topic'],
                    key=f"{result['camera_id']}_{result['frame_id']}_{result['local_vehicle_id']}".encode('utf-8'),
                    value=result
                )
            
            producer.flush()
            producer.close()
            
            logger.info(f"Sent {len(results)} results to Kafka topic {KAFKA_CONFIG['reid_results_topic']}")
            
        except Exception as e:
            logger.error(f"Error sending results to Kafka: {e}")
    
    def _save_state(self):
        """Save the current ReID state"""
        try:
            state_dir = './reid_state'
            self.cross_camera_reid.save_state(state_dir)
            
            # Save consumer stats
            stats = {
                'processed_frames': self.processed_frames,
                'total_vehicles': self.total_vehicles,
                'cross_camera_matches_count': self.cross_camera_matches_count,
                'timestamp': time.time()
            }
            
            with open(os.path.join(state_dir, 'consumer_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)
                
            logger.info("State saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def start_streaming_with_pandas_udf(self):
        """Start streaming with Pandas UDF for better performance"""
        logger.info("Starting streaming consumer with Pandas UDF...")
        
        try:
            # Read from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", ','.join(KAFKA_CONFIG['bootstrap_servers'])) \
                .option("subscribe", KAFKA_CONFIG['video_topic']) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .option("maxOffsetsPerTrigger", PROCESSING_CONFIG.get('max_offsets_per_trigger', 1000)) \
                .load()
            
            # Parse Kafka messages
            parsed_df = df.selectExpr("CAST(value AS STRING) as value", "timestamp as kafka_timestamp")
            
            # For Pandas UDF, we'll process in batches using foreachBatch
            # The Pandas UDF will be applied within the batch processing
            query = parsed_df.writeStream \
                .foreachBatch(self._process_vehicle_batch_with_pandas_udf) \
                .option("checkpointLocation", SPARK_CONFIG['checkpoint_location'] + "_pandas") \
                .outputMode("append") \
                .trigger(processingTime='15 seconds') \
                .start()
            
            logger.info("Streaming query started with Pandas UDF")
            
            # Wait for termination
            try:
                query.awaitTermination()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                query.stop()
                self._save_state()
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise
        finally:
            self.spark.stop()
            logger.info("Spark session stopped")
    
    def _process_vehicle_batch_with_pandas_udf(self, df: DataFrame, epoch_id: int):
        """Process a batch using Pandas UDF for better performance"""
        try:
            logger.info(f"Processing batch {epoch_id} with Pandas UDF")
            
            if df.count() == 0:
                logger.info(f"Batch {epoch_id}: No data to process")
                return
            
            # Apply Pandas UDF processing directly on Spark DataFrame
            # First, create a temporary view or use the DataFrame directly
            temp_view_name = f"kafka_messages_batch_{epoch_id}"
            df.createOrReplaceTempView(temp_view_name)
            
            # Use SQL to apply the UDF if needed, or use DataFrame operations
            # For now, let's process the data in a more compatible way
            parsed_df = df.selectExpr("CAST(value AS STRING) as value")
            
            # Apply our processing function (not as UDF for now, but as batch processing)
            pandas_df = parsed_df.toPandas()
            
            if pandas_df.empty:
                logger.info(f"Batch {epoch_id}: No data to process after conversion")
                return
            
            # Process using our function directly (not as UDF)
            result_df = self._process_pandas_batch_direct(pandas_df)
            
            if result_df.empty:
                logger.info(f"Batch {epoch_id}: No results from processing")
                return
            
            # Convert results to Kafka format
            kafka_results = []
            for _, row in result_df.iterrows():
                result = {
                    'timestamp': row['timestamp'],
                    'camera_id': int(row['camera_id']),
                    'frame_id': int(row['frame_id']),
                    'local_vehicle_id': row['local_vehicle_id'],
                    'global_vehicle_id': row['global_vehicle_id'],
                    'vehicle_type': row['vehicle_type'],
                    'bbox': {
                        'xtl': float(row['bbox_xtl']),
                        'ytl': float(row['bbox_ytl']),
                        'xbr': float(row['bbox_xbr']),
                        'ybr': float(row['bbox_ybr'])
                    },
                    'confidence': float(row['confidence']),
                    'features_available': len(row['features']) > 0
                }
                kafka_results.append(result)
                self.total_vehicles += 1
            
            # Send to Kafka
            if kafka_results:
                self._send_results_to_kafka(kafka_results)
            
            # Update statistics
            self.processed_frames += len(set((r['camera_id'], r['frame_id']) for r in kafka_results))
            
            # Save state periodically
            if self.processed_frames % 50 == 0:
                self._save_state()
            
            logger.info(f"Pandas UDF batch {epoch_id} completed: {len(kafka_results)} vehicles processed")
            
        except Exception as e:
            logger.error(f"Error in Pandas UDF batch processing {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_pandas_batch_direct(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """Process a pandas DataFrame directly without UDF wrapper"""
        try:
            # Initialize ReID model for this batch
            reid_model, cross_camera_reid = get_reid_model()
            results = []
            
            logger.info(f"Processing batch of {len(pandas_df)} messages directly")
            
            # Group messages by camera for efficient processing
            camera_groups = {}
            for idx, row in pandas_df.iterrows():
                try:
                    message_dict = json.loads(row['value'])
                    if message_dict.get('message_type') != 'frame_with_annotations':
                        continue
                    
                    camera_id = message_dict['camera_id']
                    if camera_id not in camera_groups:
                        camera_groups[camera_id] = []
                    camera_groups[camera_id].append(message_dict)
                    
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    continue
            
            # Process each camera group
            for camera_id, messages in camera_groups.items():
                try:
                    # Sort messages by frame_id for temporal consistency
                    messages.sort(key=lambda x: x.get('frame_id', 0))
                    
                    for message_dict in messages:
                        try:
                            # Decode frame and annotations
                            frame, annotations = decode_kafka_message(message_dict)
                            
                            frame_id = message_dict['frame_id']
                            timestamp = message_dict['timestamp']
                            
                            # Batch process all vehicles in the frame
                            vehicle_crops = []
                            annotation_data = []
                            
                            for annotation in annotations:
                                x1, y1 = int(annotation.xtl), int(annotation.ytl)
                                x2, y2 = int(annotation.xbr), int(annotation.ybr)
                                
                                if x2 > x1 and y2 > y1:
                                    vehicle_crop = frame[y1:y2, x1:x2]
                                    vehicle_crops.append(vehicle_crop)
                                    annotation_data.append(annotation)
                            
                            # Batch extract features for all vehicles in the frame
                            if vehicle_crops:
                                batch_features = reid_model.extract_features_batch(vehicle_crops)
                                
                                # Process each vehicle with its features
                                for i, (annotation, features) in enumerate(zip(annotation_data, batch_features)):
                                    try:
                                        # Perform ReID
                                        global_id = cross_camera_reid.process_vehicle(
                                            vehicle_crop=vehicle_crops[i],
                                            vehicle_id=annotation.vehicle_id,
                                            camera_id=camera_id,
                                            vehicle_type=annotation.label,
                                            frame_id=frame_id
                                        )
                                        
                                        # Encode features as base64 for serialization
                                        features_b64 = base64.b64encode(features.tobytes()).decode('utf-8')
                                        
                                        # Create result
                                        result = {
                                            'timestamp': timestamp,
                                            'camera_id': camera_id,
                                            'frame_id': frame_id,
                                            'local_vehicle_id': annotation.vehicle_id,
                                            'global_vehicle_id': global_id,
                                            'vehicle_type': annotation.label,
                                            'bbox_xtl': float(annotation.xtl),
                                            'bbox_ytl': float(annotation.ytl),
                                            'bbox_xbr': float(annotation.xbr),
                                            'bbox_ybr': float(annotation.ybr),
                                            'confidence': float(annotation.confidence),
                                            'features': features_b64
                                        }
                                        
                                        results.append(result)
                                        
                                    except Exception as e:
                                        logger.error(f"Error processing vehicle {annotation.vehicle_id}: {e}")
                                        continue
                            
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_id}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error processing camera {camera_id}: {e}")
                    continue
            
            # Convert results to DataFrame
            if results:
                result_df = pd.DataFrame(results)
                logger.info(f"Direct batch processed {len(results)} vehicles from {len(camera_groups)} cameras")
                return result_df
            else:
                # Return empty DataFrame with correct schema
                empty_df = pd.DataFrame(columns=[
                    'timestamp', 'camera_id', 'frame_id', 'local_vehicle_id', 'global_vehicle_id',
                    'vehicle_type', 'bbox_xtl', 'bbox_ytl', 'bbox_xbr', 'bbox_ybr', 'confidence', 'features'
                ])
                logger.info("Direct batch processed 0 vehicles")
                return empty_df
                
        except Exception as e:
            logger.error(f"Error in direct batch processing: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'camera_id', 'frame_id', 'local_vehicle_id', 'global_vehicle_id',
                'vehicle_type', 'bbox_xtl', 'bbox_ytl', 'bbox_xbr', 'bbox_ybr', 'confidence', 'features'
            ])
            return empty_df
    
    def _send_pandas_results_to_kafka(self, df: DataFrame, epoch_id: int):
        """Send Pandas UDF results to Kafka"""
        try:
            logger.info(f"Processing Pandas UDF batch {epoch_id}")
            
            # Collect results
            results = df.collect()
            
            if not results:
                logger.info(f"Batch {epoch_id}: No results to send")
                return
            
            # Convert to standard format for Kafka
            kafka_results = []
            for row in results:
                # Decode features if needed
                features_b64 = row['features']
                if features_b64:
                    try:
                        features = np.frombuffer(base64.b64decode(features_b64), dtype=np.float32)
                    except:
                        features = None
                else:
                    features = None
                
                result = {
                    'timestamp': row['timestamp'],
                    'camera_id': row['camera_id'],
                    'frame_id': row['frame_id'],
                    'local_vehicle_id': row['local_vehicle_id'],
                    'global_vehicle_id': row['global_vehicle_id'],
                    'vehicle_type': row['vehicle_type'],
                    'bbox': {
                        'xtl': row['bbox_xtl'],
                        'ytl': row['bbox_ytl'],
                        'xbr': row['bbox_xbr'],
                        'ybr': row['bbox_ybr']
                    },
                    'confidence': row['confidence'],
                    'features_available': features is not None
                }
                
                kafka_results.append(result)
                self.total_vehicles += 1
            
            # Send to Kafka
            if kafka_results:
                self._send_results_to_kafka(kafka_results)
            
            # Update statistics
            self.processed_frames += len(set((r['camera_id'], r['frame_id']) for r in kafka_results))
            
            # Save state periodically
            if self.processed_frames % 50 == 0:
                self._save_state()
            
            logger.info(f"Pandas UDF batch {epoch_id} completed: {len(kafka_results)} vehicles processed")
            
        except Exception as e:
            logger.error(f"Error in Pandas UDF batch processing {epoch_id}: {e}")
    
    def start_streaming(self):
        """Start the streaming consumer - choose between different processing approaches"""
        use_pandas_processing = PROCESSING_CONFIG.get('enable_pandas_udf', True) and PANDAS_UDF_AVAILABLE
        
        if use_pandas_processing:
            logger.info("Using efficient Pandas batch processing")
            self.start_streaming_with_efficient_pandas()
        else:
            logger.info("Using regular batch processing")
            self.start_streaming_regular()
    
    def start_streaming_with_efficient_pandas(self):
        """Start streaming with efficient Pandas batch processing (no UDFs)"""
        logger.info("Starting streaming consumer with efficient Pandas processing...")
        
        try:
            # Read from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", ','.join(KAFKA_CONFIG['bootstrap_servers'])) \
                .option("subscribe", KAFKA_CONFIG['video_topic']) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .option("maxOffsetsPerTrigger", PROCESSING_CONFIG.get('max_offsets_per_trigger', 500)) \
                .load()
            
            # Start the streaming query with efficient batch processing
            query = df.writeStream \
                .foreachBatch(self._process_vehicle_batch_efficient_pandas) \
                .option("checkpointLocation", SPARK_CONFIG['checkpoint_location'] + "_efficient_pandas") \
                .outputMode("append") \
                .trigger(processingTime='10 seconds') \
                .start()
            
            logger.info("Streaming query started with efficient Pandas processing")
            
            # Wait for termination
            try:
                query.awaitTermination()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                query.stop()
                self._save_state()
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise
        finally:
            self.spark.stop()
            logger.info("Spark session stopped")
    
    def _process_vehicle_batch_efficient_pandas(self, df: DataFrame, epoch_id: int):
        """Process a batch using efficient Pandas processing without UDFs"""
        try:
            logger.info(f"Processing batch {epoch_id} with efficient Pandas processing")
            
            if df.count() == 0:
                logger.info(f"Batch {epoch_id}: No data to process")
                return
            
            # Convert Spark DataFrame to Pandas for processing
            pandas_df = df.selectExpr("CAST(value AS STRING) as value").toPandas()
            
            if pandas_df.empty:
                logger.info(f"Batch {epoch_id}: No data to process after conversion")
                return
            
            # Process using efficient batch method
            batch_results = self._process_pandas_batch_efficient(pandas_df)
            
            # Send results to Kafka
            if batch_results:
                self._send_results_to_kafka(batch_results)
            
            # Update statistics
            self.processed_frames += len(set((r['camera_id'], r['frame_id']) for r in batch_results))
            self.total_vehicles += len(batch_results)
            
            # Save state periodically
            if self.processed_frames % 50 == 0:
                self._save_state()
            
            logger.info(f"Efficient Pandas batch {epoch_id} completed: {len(batch_results)} vehicles processed")
            
        except Exception as e:
            logger.error(f"Error in efficient Pandas batch processing {epoch_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_pandas_batch_efficient(self, pandas_df: pd.DataFrame) -> List[Dict]:
        """Process a pandas DataFrame efficiently with batch feature extraction"""
        try:
            # Initialize ReID model for this batch
            reid_model, cross_camera_reid = get_reid_model()
            results = []
            
            logger.info(f"Processing batch of {len(pandas_df)} messages efficiently")
            
            # Group messages by camera and frame for efficient processing
            frame_groups = {}
            for idx, row in pandas_df.iterrows():
                try:
                    message_dict = json.loads(row['value'])
                    if message_dict.get('message_type') != 'frame_with_annotations':
                        continue
                    
                    camera_id = message_dict['camera_id']
                    frame_id = message_dict['frame_id']
                    key = (camera_id, frame_id)
                    
                    if key not in frame_groups:
                        frame_groups[key] = []
                    frame_groups[key].append(message_dict)
                    
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    continue
            
            # Process each frame group
            for (camera_id, frame_id), messages in frame_groups.items():
                try:
                    # Take the first message for frame data (they should be the same)
                    message_dict = messages[0]
                    
                    # Decode frame and annotations
                    frame, annotations = decode_kafka_message(message_dict)
                    timestamp = message_dict['timestamp']
                    
                    if not annotations:
                        continue
                    
                    # Extract all vehicle crops for batch processing
                    vehicle_crops = []
                    annotation_data = []
                    
                    for annotation in annotations:
                        x1, y1 = int(annotation.xtl), int(annotation.ytl)
                        x2, y2 = int(annotation.xbr), int(annotation.ybr)
                        
                        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                            # Ensure crop is within frame bounds
                            h, w = frame.shape[:2]
                            x1 = builtins.max(0, int(x1))
                            y1 = builtins.max(0, int(y1))
                            x2 = builtins.min(w, int(x2))
                            y2 = builtins.min(h, int(y2))
                            
                            if x2 > x1 and y2 > y1:
                                vehicle_crop = frame[y1:y2, x1:x2]
                                if vehicle_crop.size > 0:
                                    vehicle_crops.append(vehicle_crop)
                                    annotation_data.append(annotation)
                    
                    # Batch extract features for all vehicles in the frame
                    if vehicle_crops:
                        batch_features = reid_model.extract_features_batch(vehicle_crops)
                        
                        # Process each vehicle with its features
                        for i, (annotation, features) in enumerate(zip(annotation_data, batch_features)):
                            try:
                                # Perform ReID
                                global_id = cross_camera_reid.process_vehicle(
                                    vehicle_crop=vehicle_crops[i],
                                    vehicle_id=annotation.vehicle_id,
                                    camera_id=camera_id,
                                    vehicle_type=annotation.label,
                                    frame_id=frame_id
                                )
                                
                                # Create result (without heavy feature serialization)
                                result = {
                                    'timestamp': timestamp,
                                    'camera_id': camera_id,
                                    'frame_id': frame_id,
                                    'local_vehicle_id': annotation.vehicle_id,
                                    'global_vehicle_id': global_id,
                                    'vehicle_type': annotation.label,
                                    'bbox': {
                                        'xtl': annotation.xtl,
                                        'ytl': annotation.ytl,
                                        'xbr': annotation.xbr,
                                        'ybr': annotation.ybr
                                    },
                                    'confidence': annotation.confidence
                                }
                                
                                results.append(result)
                                
                            except Exception as e:
                                logger.error(f"Error processing vehicle {annotation.vehicle_id}: {e}")
                                continue
                    
                except Exception as e:
                    logger.error(f"Error processing frame ({camera_id}, {frame_id}): {e}")
                    continue
            
            logger.info(f"Efficient batch processed {len(results)} vehicles from {len(frame_groups)} frames")
            return results
                
        except Exception as e:
            logger.error(f"Error in efficient batch processing: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def start_streaming_regular(self):
        """Start the streaming consumer with regular processing"""
        logger.info("Starting streaming consumer with regular processing...")
        
        try:
            # Read from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", ','.join(KAFKA_CONFIG['bootstrap_servers'])) \
                .option("subscribe", KAFKA_CONFIG['video_topic']) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .load()
            
            # Start the streaming query
            query = df.writeStream \
                .foreachBatch(self._process_vehicle_batch) \
                .option("checkpointLocation", SPARK_CONFIG['checkpoint_location']) \
                .outputMode("append") \
                .trigger(processingTime='10 seconds') \
                .start()
            
            logger.info("Streaming query started")
            
            # Wait for termination
            try:
                query.awaitTermination()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                query.stop()
                self._save_state()
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise
        finally:
            self.spark.stop()
            logger.info("Spark session stopped")
        """Start the streaming consumer"""
        logger.info("Starting streaming consumer...")
        
        try:
            # Read from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", ','.join(KAFKA_CONFIG['bootstrap_servers'])) \
                .option("subscribe", KAFKA_CONFIG['video_topic']) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .load()
            
            # Start the streaming query
            query = df.writeStream \
                .foreachBatch(self._process_vehicle_batch) \
                .option("checkpointLocation", SPARK_CONFIG['checkpoint_location']) \
                .outputMode("append") \
                .trigger(processingTime='10 seconds') \
                .start()
            
            logger.info("Streaming query started")
            
            # Wait for termination
            try:
                query.awaitTermination()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                query.stop()
                self._save_state()
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise
        finally:
            self.spark.stop()
            logger.info("Spark session stopped")
    
    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        gallery_stats = self.reid_model.get_gallery_stats()
        cross_camera_matches = self.cross_camera_reid.get_cross_camera_matches()
        
        stats = {
            'processed_frames': self.processed_frames,
            'total_vehicles': self.total_vehicles,
            'cross_camera_matches_count': self.cross_camera_matches_count,
            'gallery_stats': gallery_stats,
            'cross_camera_matches': {
                str(key): len(matches) for key, matches in cross_camera_matches.items()
            },
            'global_id_mapping_count': len(self.cross_camera_reid.get_global_id_mapping())
        }
        
        return stats

# Global ReID model instance for UDF
_reid_model = None
_cross_camera_reid = None

def get_reid_model():
    """Get or create global ReID model instance"""
    global _reid_model, _cross_camera_reid
    if _reid_model is None:
        _reid_model = VehicleReIDModel(
            model_name=REID_CONFIG['model_name'],
            pretrained=REID_CONFIG['pretrained'],
            feature_dim=REID_CONFIG['feature_dim']
        )
        _cross_camera_reid = CrossCameraReID(
            reid_model=_reid_model,
            similarity_threshold=REID_CONFIG['similarity_threshold']
        )
        
        # Try to load existing state
        state_dir = './reid_state'
        if os.path.exists(state_dir):
            _cross_camera_reid.load_state(state_dir)
            logger.info("Loaded existing ReID state in UDF")
    
    return _reid_model, _cross_camera_reid

# Define output schema for Pandas UDF
vehicle_reid_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("camera_id", IntegerType(), True),
    StructField("frame_id", IntegerType(), True),
    StructField("local_vehicle_id", StringType(), True),
    StructField("global_vehicle_id", StringType(), True),
    StructField("vehicle_type", StringType(), True),
    StructField("bbox_xtl", DoubleType(), True),
    StructField("bbox_ytl", DoubleType(), True),
    StructField("bbox_xbr", DoubleType(), True),
    StructField("bbox_ybr", DoubleType(), True),
    StructField("confidence", DoubleType(), True),
    StructField("features", StringType(), True)
])

if PANDAS_UDF_AVAILABLE:
    def process_vehicle_reid_pandas_iter(iterator):
        """Iterator-based Pandas UDF for processing vehicle re-identification"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Initialize ReID model for this partition
        reid_model, cross_camera_reid = get_reid_model()
        
        for pandas_df in iterator:
            try:
                results = []
                
                logger.info(f"Processing pandas batch of {len(pandas_df)} messages")
                
                # Group messages by camera for efficient processing
                camera_groups = {}
                for idx, row in pandas_df.iterrows():
                    try:
                        message_dict = json.loads(row['value'])
                        if message_dict.get('message_type') != 'frame_with_annotations':
                            continue
                        
                        camera_id = message_dict['camera_id']
                        if camera_id not in camera_groups:
                            camera_groups[camera_id] = []
                        camera_groups[camera_id].append(message_dict)
                        
                    except Exception as e:
                        logger.error(f"Error parsing message: {e}")
                        continue
                
                # Process each camera group
                for camera_id, messages in camera_groups.items():
                    try:
                        # Sort messages by frame_id for temporal consistency
                        messages.sort(key=lambda x: x.get('frame_id', 0))
                        
                        for message_dict in messages:
                            try:
                                # Decode frame and annotations
                                frame, annotations = decode_kafka_message(message_dict)
                                
                                frame_id = message_dict['frame_id']
                                timestamp = message_dict['timestamp']
                                
                                # Batch process all vehicles in the frame
                                vehicle_crops = []
                                annotation_data = []
                                
                                for annotation in annotations:
                                    x1, y1 = int(annotation.xtl), int(annotation.ytl)
                                    x2, y2 = int(annotation.xbr), int(annotation.ybr)
                                    
                                    if x2 > x1 and y2 > y1:
                                        vehicle_crop = frame[y1:y2, x1:x2]
                                        vehicle_crops.append(vehicle_crop)
                                        annotation_data.append(annotation)
                                
                                # Batch extract features for all vehicles in the frame
                                if vehicle_crops:
                                    batch_features = reid_model.extract_features_batch(vehicle_crops)
                                    
                                    # Process each vehicle with its features
                                    for i, (annotation, features) in enumerate(zip(annotation_data, batch_features)):
                                        try:
                                            # Perform ReID
                                            global_id = cross_camera_reid.process_vehicle(
                                                vehicle_crop=vehicle_crops[i],
                                                vehicle_id=annotation.vehicle_id,
                                                camera_id=camera_id,
                                                vehicle_type=annotation.label,
                                                frame_id=frame_id
                                            )
                                            
                                            # Encode features as base64 for serialization
                                            features_b64 = base64.b64encode(features.tobytes()).decode('utf-8')
                                            
                                            # Create result
                                            result = {
                                                'timestamp': timestamp,
                                                'camera_id': camera_id,
                                                'frame_id': frame_id,
                                                'local_vehicle_id': annotation.vehicle_id,
                                                'global_vehicle_id': global_id,
                                                'vehicle_type': annotation.label,
                                                'bbox_xtl': float(annotation.xtl),
                                                'bbox_ytl': float(annotation.ytl),
                                                'bbox_xbr': float(annotation.xbr),
                                                'bbox_ybr': float(annotation.ybr),
                                                'confidence': float(annotation.confidence),
                                                'features': features_b64
                                            }
                                            
                                            results.append(result)
                                            
                                        except Exception as e:
                                            logger.error(f"Error processing vehicle {annotation.vehicle_id}: {e}")
                                            continue
                                
                            except Exception as e:
                                logger.error(f"Error processing frame {frame_id}: {e}")
                                continue
                        
                    except Exception as e:
                        logger.error(f"Error processing camera {camera_id}: {e}")
                        continue
                
                # Convert results to DataFrame
                if results:
                    result_df = pd.DataFrame(results)
                    logger.info(f"Pandas iterator processed {len(results)} vehicles from {len(camera_groups)} cameras")
                    yield result_df
                else:
                    # Return empty DataFrame with correct schema
                    empty_df = pd.DataFrame(columns=[
                        'timestamp', 'camera_id', 'frame_id', 'local_vehicle_id', 'global_vehicle_id',
                        'vehicle_type', 'bbox_xtl', 'bbox_ytl', 'bbox_xbr', 'bbox_ybr', 'confidence', 'features'
                    ])
                    logger.info("Pandas iterator processed 0 vehicles")
                    yield empty_df
                    
            except Exception as e:
                logger.error(f"Error in Pandas iterator processing: {e}")
                import traceback
                traceback.print_exc()
                # Return empty DataFrame with correct schema
                empty_df = pd.DataFrame(columns=[
                    'timestamp', 'camera_id', 'frame_id', 'local_vehicle_id', 'global_vehicle_id',
                    'vehicle_type', 'bbox_xtl', 'bbox_ytl', 'bbox_xbr', 'bbox_ybr', 'confidence', 'features'
                ])
                yield empty_df

    def start_streaming_with_map_in_pandas(self):
        """Start streaming with mapInPandas for efficient batch processing"""
        logger.info("Starting streaming consumer with mapInPandas...")
        
        try:
            # Read from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", ','.join(KAFKA_CONFIG['bootstrap_servers'])) \
                .option("subscribe", KAFKA_CONFIG['video_topic']) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .option("maxOffsetsPerTrigger", PROCESSING_CONFIG.get('max_offsets_per_trigger', 1000)) \
                .load()
            
            # Parse Kafka messages
            parsed_df = df.selectExpr("CAST(value AS STRING) as value")
            
            # Apply mapInPandas for efficient batch processing
            results_df = parsed_df.mapInPandas(
                process_vehicle_reid_pandas_iter,
                schema=vehicle_reid_schema
            )
            
            # Start the streaming query
            query = results_df.writeStream \
                .foreachBatch(self._send_pandas_results_to_kafka) \
                .option("checkpointLocation", SPARK_CONFIG['checkpoint_location'] + "_map_pandas") \
                .outputMode("append") \
                .trigger(processingTime='15 seconds') \
                .start()
            
            logger.info("Streaming query started with mapInPandas")
            
            # Wait for termination
            try:
                query.awaitTermination()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                query.stop()
                self._save_state()
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise
        finally:
            self.spark.stop()
            logger.info("Spark session stopped")

    # Update the main streaming method to use mapInPandas
    VehicleReIDSparkConsumer.start_streaming_with_map_in_pandas = start_streaming_with_map_in_pandas
    @pandas_udf(returnType=vehicle_reid_schema)
    def process_vehicle_reid_batch(pdf: pd.DataFrame) -> pd.DataFrame:
        """Pandas UDF for processing vehicle re-identification in batches"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Initialize ReID model for this worker
            reid_model, cross_camera_reid = get_reid_model()
            results = []
            
            logger.info(f"Processing batch of {len(pdf)} messages with Pandas UDF")
            
            for idx, row in pdf.iterrows():
                try:
                    # Parse the Kafka message
                    message_dict = json.loads(row['value'])
                    
                    if message_dict.get('message_type') != 'frame_with_annotations':
                        continue
                    
                    # Decode frame and annotations
                    frame, annotations = decode_kafka_message(message_dict)
                    
                    camera_id = message_dict['camera_id']
                    frame_id = message_dict['frame_id']
                    timestamp = message_dict['timestamp']
                    
                    # Process each vehicle in the frame
                    for annotation in annotations:
                        try:
                            # Extract vehicle crop
                            x1, y1 = int(annotation.xtl), int(annotation.ytl)
                            x2, y2 = int(annotation.xbr), int(annotation.ybr)
                            
                            if x2 > x1 and y2 > y1:
                                vehicle_crop = frame[y1:y2, x1:x2]
                                
                                # Perform ReID
                                global_id = cross_camera_reid.process_vehicle(
                                    vehicle_crop=vehicle_crop,
                                    vehicle_id=annotation.vehicle_id,
                                    camera_id=camera_id,
                                    vehicle_type=annotation.label,
                                    frame_id=frame_id
                                )
                                
                                # Extract features for additional processing
                                features = reid_model.extract_features(vehicle_crop)
                                
                                # Encode features as base64 for serialization
                                features_b64 = base64.b64encode(features.tobytes()).decode('utf-8')
                                
                                # Create result
                                result = {
                                    'timestamp': timestamp,
                                    'camera_id': camera_id,
                                    'frame_id': frame_id,
                                    'local_vehicle_id': annotation.vehicle_id,
                                    'global_vehicle_id': global_id,
                                    'vehicle_type': annotation.label,
                                    'bbox_xtl': float(annotation.xtl),
                                    'bbox_ytl': float(annotation.ytl),
                                    'bbox_xbr': float(annotation.xbr),
                                    'bbox_ybr': float(annotation.ybr),
                                    'confidence': float(annotation.confidence),
                                    'features': features_b64
                                }
                                
                                results.append(result)
                                
                        except Exception as e:
                            logger.error(f"Error processing vehicle {annotation.vehicle_id}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
            
            # Convert results to DataFrame
            if results:
                result_df = pd.DataFrame(results)
                logger.info(f"Pandas UDF processed {len(results)} vehicles")
                return result_df
            else:
                # Return empty DataFrame with correct schema
                empty_df = pd.DataFrame(columns=[field.name for field in vehicle_reid_schema.fields])
                logger.info("Pandas UDF processed 0 vehicles")
                return empty_df
                
        except Exception as e:
            logger.error(f"Error in Pandas UDF: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=[field.name for field in vehicle_reid_schema.fields])
            return empty_df

    @pandas_udf(returnType=vehicle_reid_schema)
    def process_vehicle_features_batch(pdf: pd.DataFrame) -> pd.DataFrame:
        """Alternative Pandas UDF focusing on feature extraction efficiency"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Initialize ReID model for this worker
            reid_model, cross_camera_reid = get_reid_model()
            results = []
            
            # Group messages by camera for efficient processing
            camera_groups = {}
            for idx, row in pdf.iterrows():
                try:
                    message_dict = json.loads(row['value'])
                    if message_dict.get('message_type') != 'frame_with_annotations':
                        continue
                    
                    camera_id = message_dict['camera_id']
                    if camera_id not in camera_groups:
                        camera_groups[camera_id] = []
                    camera_groups[camera_id].append(message_dict)
                    
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    continue
            
            # Process each camera group
            for camera_id, messages in camera_groups.items():
                try:
                    # Sort messages by frame_id for temporal consistency
                    messages.sort(key=lambda x: x.get('frame_id', 0))
                    
                    for message_dict in messages:
                        try:
                            # Decode frame and annotations
                            frame, annotations = decode_kafka_message(message_dict)
                            
                            frame_id = message_dict['frame_id']
                            timestamp = message_dict['timestamp']
                            
                            # Batch process all vehicles in the frame
                            vehicle_crops = []
                            annotation_data = []
                            
                            for annotation in annotations:
                                x1, y1 = int(annotation.xtl), int(annotation.ytl)
                                x2, y2 = int(annotation.xbr), int(annotation.ybr)
                                
                                if x2 > x1 and y2 > y1:
                                    vehicle_crop = frame[y1:y2, x1:x2]
                                    vehicle_crops.append(vehicle_crop)
                                    annotation_data.append(annotation)
                            
                            # Batch extract features for all vehicles in the frame
                            if vehicle_crops:
                                batch_features = reid_model.extract_features_batch(vehicle_crops)
                                
                                # Process each vehicle with its features
                                for i, (annotation, features) in enumerate(zip(annotation_data, batch_features)):
                                    try:
                                        # Perform ReID
                                        global_id = cross_camera_reid.process_vehicle(
                                            vehicle_crop=vehicle_crops[i],
                                            vehicle_id=annotation.vehicle_id,
                                            camera_id=camera_id,
                                            vehicle_type=annotation.label,
                                            frame_id=frame_id
                                        )
                                        
                                        # Encode features as base64 for serialization
                                        features_b64 = base64.b64encode(features.tobytes()).decode('utf-8')
                                        
                                        # Create result
                                        result = {
                                            'timestamp': timestamp,
                                            'camera_id': camera_id,
                                            'frame_id': frame_id,
                                            'local_vehicle_id': annotation.vehicle_id,
                                            'global_vehicle_id': global_id,
                                            'vehicle_type': annotation.label,
                                            'bbox_xtl': float(annotation.xtl),
                                            'bbox_ytl': float(annotation.ytl),
                                            'bbox_xbr': float(annotation.xbr),
                                            'bbox_ybr': float(annotation.ybr),
                                            'confidence': float(annotation.confidence),
                                            'features': features_b64
                                        }
                                        
                                        results.append(result)
                                        
                                    except Exception as e:
                                        logger.error(f"Error processing vehicle {annotation.vehicle_id}: {e}")
                                        continue
                            
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_id}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error processing camera {camera_id}: {e}")
                    continue
            
            # Convert results to DataFrame
            if results:
                result_df = pd.DataFrame(results)
                logger.info(f"Pandas UDF batch processed {len(results)} vehicles from {len(camera_groups)} cameras")
                return result_df
            else:
                # Return empty DataFrame with correct schema
                empty_df = pd.DataFrame(columns=[field.name for field in vehicle_reid_schema.fields])
                logger.info("Pandas UDF batch processed 0 vehicles")
                return empty_df
                
        except Exception as e:
            logger.error(f"Error in Pandas UDF batch processing: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame with correct schema
            empty_df = pd.DataFrame(columns=[field.name for field in vehicle_reid_schema.fields])
            return empty_df

else:
    # Fallback function for when Pandas UDF is not available
    def process_vehicle_reid_batch(pdf: pd.DataFrame) -> pd.DataFrame:
        logger.warning("Pandas UDF not available, using fallback processing")
        return pd.DataFrame(columns=[field.name for field in vehicle_reid_schema.fields])
    
    def process_vehicle_features_batch(pdf: pd.DataFrame) -> pd.DataFrame:
        logger.warning("Pandas UDF not available, using fallback processing")
        return pd.DataFrame(columns=[field.name for field in vehicle_reid_schema.fields])

def main():
    """Main function to run the consumer"""
    logger.info("Starting Vehicle ReID Spark Consumer")
    
    consumer = VehicleReIDSparkConsumer()
    
    try:
        # Display initial stats
        stats = consumer.get_stats()
        logger.info("Consumer Statistics:")
        logger.info(f"  Gallery vehicles: {stats['gallery_stats']['total_vehicles']}")
        logger.info(f"  Gallery features: {stats['gallery_stats']['total_features']}")
        
        # Start streaming
        consumer.start_streaming()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Consumer shutdown complete")

if __name__ == "__main__":
    main()
