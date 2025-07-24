#!/bin/bash

# Setup script for Kafka environment
echo "Setting up Kafka environment for Vehicle ReID system..."

# Check if Kafka is already running
check_kafka_running() {
    if pgrep -f "kafka.Kafka" > /dev/null; then
        echo "Kafka is already running"
        return 0
    else
        return 1
    fi
}

check_zookeeper_running() {
    if pgrep -f "zookeeper" > /dev/null; then
        echo "Zookeeper is already running"
        return 0
    else
        return 1
    fi
}

# Start Zookeeper if not running
if ! check_zookeeper_running; then
    echo "Starting Zookeeper..."
    cd ../reid
    if [ -f "zookeeper.pid" ]; then
        echo "Removing old Zookeeper PID file..."
        rm zookeeper.pid
    fi
    
    # Start Zookeeper in background
    nohup bash start_kafka.sh > zookeeper.log 2>&1 &
    sleep 5
    
    if check_zookeeper_running; then
        echo "Zookeeper started successfully"
    else
        echo "Failed to start Zookeeper"
        exit 1
    fi
fi

# Start Kafka if not running
if ! check_kafka_running; then
    echo "Starting Kafka..."
    cd ../reid
    if [ -f "kafka.pid" ]; then
        echo "Removing old Kafka PID file..."
        rm kafka.pid
    fi
    
    # Start Kafka in background
    nohup bash start_kafka.sh > kafka.log 2>&1 &
    sleep 10
    
    if check_kafka_running; then
        echo "Kafka started successfully"
    else
        echo "Failed to start Kafka"
        exit 1
    fi
fi

# Return to new_reid_method directory
cd ../new_reid_method

# Create Kafka topics
echo "Creating Kafka topics..."

# Function to create topic if it doesn't exist
create_topic_if_not_exists() {
    topic_name=$1
    partitions=${2:-3}
    replication=${3:-1}
    
    # Check if topic exists
    if kafka-topics --bootstrap-server localhost:9092 --list | grep -q "^${topic_name}$"; then
        echo "Topic '${topic_name}' already exists"
    else
        echo "Creating topic '${topic_name}'..."
        kafka-topics --create \
            --bootstrap-server localhost:9092 \
            --replication-factor ${replication} \
            --partitions ${partitions} \
            --topic ${topic_name}
        
        if [ $? -eq 0 ]; then
            echo "Topic '${topic_name}' created successfully"
        else
            echo "Failed to create topic '${topic_name}'"
        fi
    fi
}

# Create topics
create_topic_if_not_exists "vehicle_video_frames" 3 1
create_topic_if_not_exists "vehicle_detections" 3 1
create_topic_if_not_exists "reid_results" 3 1

# List all topics to verify
echo "Available Kafka topics:"
kafka-topics --bootstrap-server localhost:9092 --list

# Create output directories
echo "Creating output directories..."
mkdir -p output
mkdir -p reid_state
mkdir -p checkpoint

echo "Setup complete!"
echo ""
echo "To run the Vehicle ReID system:"
echo "1. Terminal 1: python producer.py"
echo "2. Terminal 2: python consumer.py"
echo "3. Terminal 3: python visualizer.py"
echo ""
echo "To stop Kafka:"
echo "bash ../reid/stop_kafka.sh"
