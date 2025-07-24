#!/usr/bin/env python3
"""
Test script for Pandas UDF functionality
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSING_CONFIG, REID_CONFIG
from utils import AnnotationParser, VideoProcessor, create_kafka_message, decode_kafka_message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pandas_udf_processing():
    """Test Pandas UDF processing functionality"""
    logger.info("Testing Pandas UDF processing...")
    
    try:
        # Test imports
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import pandas_udf, col
            import pyspark.sql.functions as F
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType
            logger.info("✓ PySpark imports successful")
        except ImportError as e:
            logger.error(f"✗ PySpark import failed: {e}")
            return False
        
        # Test consumer import
        try:
            from consumer import VehicleReIDSparkConsumer, process_vehicle_features_batch
            logger.info("✓ Consumer imports successful")
        except ImportError as e:
            logger.error(f"✗ Consumer import failed: {e}")
            return False
        
        # Test ReID model imports
        try:
            from reid_model import VehicleReIDModel
            logger.info("✓ ReID model imports successful")
        except ImportError as e:
            logger.error(f"✗ ReID model import failed: {e}")
            return False
        
        # Test model initialization
        try:
            model = VehicleReIDModel(
                model_name=REID_CONFIG['model_name'],
                pretrained=REID_CONFIG['pretrained'],
                feature_dim=REID_CONFIG['feature_dim']
            )
            logger.info("✓ ReID model initialization successful")
        except Exception as e:
            logger.error(f"✗ ReID model initialization failed: {e}")
            return False
        
        # Test batch feature extraction
        try:
            # Create dummy vehicle crops
            dummy_crops = [
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            ]
            
            # Test batch processing
            batch_features = model.extract_features_batch(dummy_crops)
            
            assert len(batch_features) == len(dummy_crops)
            assert all(f.shape == (REID_CONFIG['feature_dim'],) for f in batch_features)
            
            logger.info("✓ Batch feature extraction successful")
        except Exception as e:
            logger.error(f"✗ Batch feature extraction failed: {e}")
            return False
        
        # Test Pandas UDF function
        try:
            # Create a sample DataFrame with Kafka message format
            sample_messages = []
            
            # Create a dummy message
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            dummy_annotations = []
            
            # Add dummy annotation
            from utils import BoundingBox
            dummy_annotation = BoundingBox(
                xtl=100, ytl=100, xbr=200, ybr=200,
                label='car', vehicle_id='test_vehicle_1',
                frame_id=0, camera_id=11, confidence=0.9
            )
            dummy_annotations.append(dummy_annotation)
            
            # Create Kafka message
            message = create_kafka_message(
                frame_data=dummy_frame,
                annotations=dummy_annotations,
                camera_id=11,
                frame_id=0,
                timestamp=datetime.now().isoformat()
            )
            
            sample_messages.append(message)
            
            # Create DataFrame for UDF testing
            test_df = pd.DataFrame({'value': [str(msg) for msg in sample_messages]})
            
            # Test UDF function
            result_df = process_vehicle_features_batch(test_df)
            
            logger.info(f"✓ Pandas UDF processing successful: {len(result_df)} results")
            
        except Exception as e:
            logger.error(f"✗ Pandas UDF processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("✓ All Pandas UDF tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pandas UDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spark_session():
    """Test Spark session creation"""
    logger.info("Testing Spark session creation...")
    
    try:
        from pyspark.sql import SparkSession
        
        # Create minimal Spark session
        spark = SparkSession.builder \
            .appName("Pandas UDF Test") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info("✓ Spark session created successfully")
        
        # Test basic DataFrame operations
        data = [("test1", 1), ("test2", 2), ("test3", 3)]
        df = spark.createDataFrame(data, ["name", "id"])
        
        assert df.count() == 3
        logger.info("✓ Basic DataFrame operations successful")
        
        # Test Pandas UDF availability
        try:
            from pyspark.sql.functions import pandas_udf
            logger.info("✓ Pandas UDF available")
        except ImportError:
            logger.warning("⚠ Pandas UDF not available")
        
        spark.stop()
        logger.info("✓ Spark session test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Spark session test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting Pandas UDF tests...")
    
    # Test Spark session
    if not test_spark_session():
        logger.error("Spark session test failed")
        return False
    
    # Test Pandas UDF processing
    if not test_pandas_udf_processing():
        logger.error("Pandas UDF processing test failed")
        return False
    
    logger.info("All tests passed! ✓")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
