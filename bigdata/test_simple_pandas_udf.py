#!/usr/bin/env python3
"""
Simplified Pandas UDF Consumer for testing
"""

import json
import logging
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a simple schema for testing
test_schema = StructType([
    StructField("input_value", StringType(), True),
    StructField("output_value", StringType(), True),
    StructField("processed_count", IntegerType(), True)
])

def create_test_spark_session():
    """Create a test Spark session"""
    spark = SparkSession.builder \
        .appName("Pandas UDF Test") \
        .master("local[2]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

@pandas_udf(returnType=test_schema)
def test_pandas_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """Simple test Pandas UDF"""
    try:
        logger.info(f"Processing {len(pdf)} rows with Pandas UDF")
        
        results = []
        for idx, row in pdf.iterrows():
            result = {
                'input_value': row['value'],
                'output_value': f"processed_{row['value']}",
                'processed_count': len(row['value']) if row['value'] else 0
            }
            results.append(result)
        
        result_df = pd.DataFrame(results)
        logger.info(f"Pandas UDF processed {len(results)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in Pandas UDF: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['input_value', 'output_value', 'processed_count'])

def main():
    """Test the Pandas UDF functionality"""
    logger.info("Starting Pandas UDF test...")
    
    # Create Spark session
    spark = create_test_spark_session()
    
    try:
        # Create test data
        test_data = [
            ("test_message_1",),
            ("test_message_2",),
            ("test_message_3",),
            ("test_message_4",),
            ("test_message_5",)
        ]
        
        # Create DataFrame
        df = spark.createDataFrame(test_data, ["value"])
        
        logger.info(f"Created test DataFrame with {df.count()} rows")
        
        # Apply Pandas UDF
        result_df = df.select(explode(array(test_pandas_udf(struct("value")))).alias("result")) \
                       .select("result.*")
        
        # Show results
        logger.info("Pandas UDF results:")
        result_df.show()
        
        # Collect results
        results = result_df.collect()
        logger.info(f"Collected {len(results)} results")
        
        for row in results:
            logger.info(f"  {row['input_value']} -> {row['output_value']} (count: {row['processed_count']})")
        
        logger.info("✓ Pandas UDF test completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Pandas UDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
