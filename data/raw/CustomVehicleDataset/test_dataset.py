#!/usr/bin/env python3
"""
Quick XML Data Validator - Just check data quality and show statistics
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

def quick_validate_xml_data():
    """Quick validation and statistics of XML data"""
    
    xml_files = {
        './annotations_11.xml': 1,  # Camera 1
        './annotations_21.xml': 2,  # Camera 2
    }
    
    all_vehicles = defaultdict(list)  # vehicle_id -> [(camera_id, count)]
    camera_stats = defaultdict(lambda: {'vehicles': set(), 'detections': 0, 'types': defaultdict(int)})
    
    print("🔍 Validating XML data...")
    
    for xml_path, camera_id in xml_files.items():
        if not Path(xml_path).exists():
            print(f"❌ {xml_path} not found!")
            continue
            
        print(f"📁 Processing {xml_path} (Camera {camera_id})")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for image in root.findall('image'):
            frame_id = int(image.get('id'))
            
            for box in image.findall('box'):
                label = box.get('label')
                
                # Get vehicle ID
                id_attr = box.find('attribute[@name="id"]')
                if id_attr is not None:
                    # Check if ID is in text content or as attribute
                    vehicle_id = id_attr.text if id_attr.text else None
                    if vehicle_id and vehicle_id.strip():  # Non-empty ID
                        all_vehicles[vehicle_id].append(camera_id)
                        camera_stats[camera_id]['vehicles'].add(vehicle_id)
                        camera_stats[camera_id]['detections'] += 1
                        camera_stats[camera_id]['types'][label] += 1
                else:
                    print(f"⚠️  Empty ID found in camera {camera_id}, frame {frame_id}")
    
    # Analysis
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    
    # Per camera stats
    for camera_id, stats in camera_stats.items():
        print(f"📷 Camera {camera_id}:")
        print(f"   Vehicles with IDs: {len(stats['vehicles'])}")
        print(f"   Total detections: {stats['detections']}")
        if 'empty_ids' in stats:
            print(f"   Detections with empty IDs: {stats['empty_ids']}")
        print(f"   Vehicle types: {dict(stats['types'])}")
    
    # Cross-camera analysis
    cross_camera_vehicles = []
    camera1_only = []
    camera2_only = []
    
    for vehicle_id, cameras in all_vehicles.items():
        camera_set = set(cameras)
        if len(camera_set) > 1:
            cross_camera_vehicles.append(vehicle_id)
        elif 1 in camera_set:
            camera1_only.append(vehicle_id)
        elif 2 in camera_set:
            camera2_only.append(vehicle_id)
    
    print(f"\n🔗 Cross-camera vehicles: {len(cross_camera_vehicles)}")
    if cross_camera_vehicles:
        print(f"   IDs: {sorted(cross_camera_vehicles)}")
    
    print(f"📷 Camera 1 only: {len(camera1_only)} vehicles")
    print(f"📷 Camera 2 only: {len(camera2_only)} vehicles")
    
    total_unique_vehicles = len(all_vehicles)
    total_detections = sum(stats['detections'] for stats in camera_stats.values())
    
    print(f"\n✅ Total unique vehicles: {total_unique_vehicles}")
    print(f"✅ Total detections: {total_detections}")
    
    # Data quality check
    print(f"\n🎯 ReID Training Strategy:")
    if cross_camera_vehicles:
        print(f"✅ {len(cross_camera_vehicles)} cross-camera pairs available for ReID training")
        print("✅ Perfect ground truth correspondences via matching IDs")
    else:
        print("📝 No cross-camera vehicles found")
        print("📝 Strategy: Use within-camera tracking + inter-camera negatives")
        print("📝 Camera 1: Needs auto-assigned IDs for tracking")
        print("📝 Camera 2: Has proper IDs for tracking")
    
    if total_unique_vehicles > 5:
        print("✅ Sufficient vehicle diversity for training")
    else:
        print("⚠️  Limited vehicle diversity - may affect model performance")
    
    print("="*50)
    
    return len(cross_camera_vehicles) > 0

if __name__ == "__main__":
    is_ready = quick_validate_xml_data()
    
    if is_ready:
        print("\n Data looks good! Ready to proceed with R-CNN feature extraction.")
    else:
        print("\nData validation found issues. Please review before proceeding.")