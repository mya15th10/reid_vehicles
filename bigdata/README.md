# Vehicle Re-Identification with Kafka and Spark

This project implements a real-time vehicle re-identification system using Apache Kafka for streaming and Apache Spark for distributed processing.

## Overview

The system processes two video feeds (Camera11 and Camera21) with annotated vehicle data to perform vehicle re-identification across different camera views.

## Components

1. **Data Processing**: XML annotation parser for CVAT format
2. **Kafka Producer**: Streams video frames and detection data
3. **Spark Consumer**: Processes streams and performs vehicle ReID
4. **Visualization**: Real-time display of re-identification results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Kafka and Zookeeper:
```bash
./setup_kafka_environment.sh
```

3. Run the system:
```bash
# Terminal 1: Start producer
python producer.py

# Terminal 2: Start consumer
python consumer.py

# Terminal 3: View results
python visualizer.py
```

## Dataset

- `video1(1).MOV` with annotations in `annotations_11.xml` (Camera11)
- `video2(1).MOV` with annotations in `annotations_21.xml` (Camera21)

## Architecture

```
Video Frames + Annotations → Kafka Producer → Kafka Topics → Spark Consumer → ReID Model → Results
```

## Features

- Real-time vehicle detection and tracking
- Cross-camera vehicle re-identification
- Distributed processing with Apache Spark
- Live visualization of results
- Support for multiple vehicle types (car, truck, bus, bicycle)
