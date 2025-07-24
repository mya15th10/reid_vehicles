"""
Analysis and evaluation tools for Vehicle ReID results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from datetime import datetime
import cv2
from collections import defaultdict, Counter

from config import CAMERA_CONFIG, DATA_PATHS
from utils import AnnotationParser
from reid_model import VehicleReIDModel, CrossCameraReID

class VehicleReIDAnalyzer:
    """Analyzer for Vehicle ReID system performance and results"""
    
    def __init__(self, results_file: str = None, reid_state_dir: str = './reid_state'):
        self.results_file = results_file
        self.reid_state_dir = reid_state_dir
        
        # Load data
        self.reid_results = []
        self.ground_truth = {}
        self.cross_camera_matches = {}
        
        if results_file and os.path.exists(results_file):
            self.load_results(results_file)
        
        # Load ground truth from annotations
        self.load_ground_truth()
        
        # Load ReID state if available
        if os.path.exists(reid_state_dir):
            self.load_reid_state()
    
    def load_results(self, results_file: str):
        """Load ReID results from JSON file"""
        with open(results_file, 'r') as f:
            self.reid_results = [json.loads(line) for line in f]
        print(f"Loaded {len(self.reid_results)} ReID results")
    
    def load_ground_truth(self):
        """Load ground truth data from annotation files"""
        for camera_key, camera_config in CAMERA_CONFIG.items():
            try:
                parser = AnnotationParser(camera_config['annotations_path'])
                tracks = parser.get_vehicle_tracks()
                self.ground_truth[camera_config['id']] = tracks
                print(f"Loaded ground truth for {camera_config['name']}: {len(tracks)} vehicles")
            except Exception as e:
                print(f"Error loading ground truth for {camera_key}: {e}")
    
    def load_reid_state(self):
        """Load ReID state for analysis"""
        import pickle
        
        state_file = os.path.join(self.reid_state_dir, 'cross_camera_state.pkl')
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                state_data = pickle.load(f)
            
            self.cross_camera_matches = state_data.get('cross_camera_matches', {})
            print(f"Loaded cross-camera matches: {len(self.cross_camera_matches)} groups")
    
    def analyze_detection_statistics(self) -> Dict:
        """Analyze vehicle detection statistics"""
        stats = {
            'total_detections': 0,
            'detections_per_camera': defaultdict(int),
            'detections_per_type': defaultdict(int),
            'detections_per_frame': defaultdict(int),
            'unique_vehicles': set(),
            'global_ids': set()
        }
        
        # Check if we have results to analyze
        if not self.reid_results:
            print("Warning: No ReID results loaded for analysis")
            stats['unique_vehicle_count'] = 0
            stats['global_id_count'] = 0
            return stats
        
        for result in self.reid_results:
            stats['total_detections'] += 1
            stats['detections_per_camera'][result['camera_id']] += 1
            stats['detections_per_type'][result['vehicle_type']] += 1
            stats['detections_per_frame'][f"{result['camera_id']}_{result['frame_id']}"] += 1
            stats['unique_vehicles'].add(result['local_vehicle_id'])
            stats['global_ids'].add(result['global_vehicle_id'])
        
        # Convert sets to counts
        stats['unique_vehicle_count'] = len(stats['unique_vehicles'])
        stats['global_id_count'] = len(stats['global_ids'])
        
        return stats
    
    def analyze_cross_camera_performance(self) -> Dict:
        """Analyze cross-camera re-identification performance"""
        # Initialize with defaults
        result = {
            'cross_camera_vehicles': [],
            'single_camera_vehicles': [],
            'cross_camera_count': 0,
            'single_camera_count': 0,
            'cross_camera_rate': 0.0
        }
        
        if not self.reid_results:
            print("Warning: No ReID results for cross-camera analysis")
            return result
        
        # Group results by global ID
        global_id_groups = defaultdict(list)
        for result_item in self.reid_results:
            global_id_groups[result_item['global_vehicle_id']].append(result_item)
        
        cross_camera_vehicles = []
        single_camera_vehicles = []
        
        for global_id, results in global_id_groups.items():
            cameras = set(r['camera_id'] for r in results)
            if len(cameras) > 1:
                cross_camera_vehicles.append({
                    'global_id': global_id,
                    'cameras': list(cameras),
                    'detections': len(results),
                    'camera_count': len(cameras)
                })
            else:
                single_camera_vehicles.append({
                    'global_id': global_id,
                    'camera': list(cameras)[0],
                    'detections': len(results)
                })
        
        result.update({
            'cross_camera_vehicles': cross_camera_vehicles,
            'single_camera_vehicles': single_camera_vehicles,
            'cross_camera_count': len(cross_camera_vehicles),
            'single_camera_count': len(single_camera_vehicles),
            'cross_camera_rate': len(cross_camera_vehicles) / len(global_id_groups) if global_id_groups else 0
        })
        
        return result
    
    def evaluate_reid_accuracy(self) -> Dict:
        """Evaluate ReID accuracy against ground truth"""
        # This is a simplified evaluation - in practice, you'd need manual annotation
        # of which vehicles are actually the same across cameras
        
        evaluation = {
            'total_ground_truth_vehicles': 0,
            'total_predicted_global_ids': 0,
            'camera_wise_accuracy': {}
        }
        
        for camera_id, tracks in self.ground_truth.items():
            evaluation['total_ground_truth_vehicles'] += len(tracks)
            
            # Get predictions for this camera
            camera_results = [r for r in self.reid_results if r['camera_id'] == camera_id]
            predicted_vehicles = set(r['local_vehicle_id'] for r in camera_results)
            ground_truth_vehicles = set(tracks.keys())
            
            # Calculate basic accuracy metrics
            true_positives = len(predicted_vehicles.intersection(ground_truth_vehicles))
            false_positives = len(predicted_vehicles - ground_truth_vehicles)
            false_negatives = len(ground_truth_vehicles - predicted_vehicles)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation['camera_wise_accuracy'][camera_id] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        
        # Count predicted global IDs
        global_ids = set(r['global_vehicle_id'] for r in self.reid_results)
        evaluation['total_predicted_global_ids'] = len(global_ids)
        
        return evaluation
    
    def create_visualizations(self, output_dir: str = './analysis_output'):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Detection statistics
        stats = self.analyze_detection_statistics()
        
        # Check if we have data to visualize
        if stats['total_detections'] == 0:
            print("Warning: No detection data to visualize. Creating placeholder charts.")
            
            # Create a simple placeholder chart
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No ReID Results Available\nRun the ReID system first to generate data', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('Vehicle Re-Identification Analysis')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'no_data_placeholder.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create ground truth visualization
            self._create_ground_truth_visualization(output_dir)
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Vehicle Detection Statistics', fontsize=16)
        
        # Detections per camera
        if stats['detections_per_camera']:
            cameras = list(stats['detections_per_camera'].keys())
            counts = list(stats['detections_per_camera'].values())
            ax1.bar([f"Camera {c}" for c in cameras], counts)
            ax1.set_title('Detections per Camera')
            ax1.set_ylabel('Number of Detections')
        else:
            ax1.text(0.5, 0.5, 'No camera data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Detections per Camera')
        
        # Detections per vehicle type
        if stats['detections_per_type']:
            types = list(stats['detections_per_type'].keys())
            type_counts = list(stats['detections_per_type'].values())
            ax2.pie(type_counts, labels=types, autopct='%1.1f%%')
            ax2.set_title('Vehicle Type Distribution')
        else:
            ax2.text(0.5, 0.5, 'No type data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Vehicle Type Distribution')
        
        # Unique vs Global IDs
        ax3.bar(['Unique Local IDs', 'Global IDs'], [stats['unique_vehicle_count'], stats['global_id_count']])
        ax3.set_title('Vehicle ID Counts')
        ax3.set_ylabel('Count')
        
        # Timeline of detections
        if self.reid_results:
            timestamps = []
            for result in self.reid_results:
                try:
                    ts = datetime.fromisoformat(result['timestamp'][:19])
                    timestamps.append(ts)
                except:
                    continue
            
            if timestamps:
                ax4.hist([ts.hour + ts.minute/60.0 for ts in timestamps], bins=20)
                ax4.set_title('Detection Timeline (Hours)')
                ax4.set_xlabel('Hour of Day')
                ax4.set_ylabel('Number of Detections')
            else:
                ax4.text(0.5, 0.5, 'No timestamp data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Detection Timeline (Hours)')
        else:
            ax4.text(0.5, 0.5, 'No timestamp data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Detection Timeline (Hours)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cross-camera analysis
        cross_camera_analysis = self.analyze_cross_camera_performance()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Cross-Camera Re-Identification Analysis', fontsize=16)
        
        # Cross-camera vs single-camera vehicles
        if cross_camera_analysis['cross_camera_count'] > 0 or cross_camera_analysis['single_camera_count'] > 0:
            labels = ['Cross-Camera', 'Single-Camera']
            sizes = [cross_camera_analysis['cross_camera_count'], cross_camera_analysis['single_camera_count']]
            # Filter out zero values to avoid pie chart issues
            non_zero_labels = []
            non_zero_sizes = []
            for label, size in zip(labels, sizes):
                if size > 0:
                    non_zero_labels.append(label)
                    non_zero_sizes.append(size)
            
            if non_zero_sizes:
                ax1.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%')
            else:
                ax1.text(0.5, 0.5, 'No vehicle data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Vehicle Distribution')
        else:
            ax1.text(0.5, 0.5, 'No vehicle data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Vehicle Distribution')
        
        # Camera coverage for cross-camera vehicles
        if cross_camera_analysis['cross_camera_vehicles']:
            camera_counts = [v['camera_count'] for v in cross_camera_analysis['cross_camera_vehicles']]
            unique_counts = list(set(camera_counts))
            count_freq = [camera_counts.count(c) for c in unique_counts]
            
            ax2.bar([f"{c} Cameras" for c in unique_counts], count_freq)
            ax2.set_title('Cross-Camera Vehicle Coverage')
            ax2.set_ylabel('Number of Vehicles')
        else:
            ax2.text(0.5, 0.5, 'No cross-camera data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Cross-Camera Vehicle Coverage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_camera_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Accuracy evaluation
        accuracy = self.evaluate_reid_accuracy()
        
        if accuracy['camera_wise_accuracy']:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('ReID Performance Evaluation', fontsize=16)
            
            cameras = list(accuracy['camera_wise_accuracy'].keys())
            precisions = [accuracy['camera_wise_accuracy'][c]['precision'] for c in cameras]
            recalls = [accuracy['camera_wise_accuracy'][c]['recall'] for c in cameras]
            f1_scores = [accuracy['camera_wise_accuracy'][c]['f1_score'] for c in cameras]
            
            x = np.arange(len(cameras))
            width = 0.25
            
            ax1.bar(x - width, precisions, width, label='Precision')
            ax1.bar(x, recalls, width, label='Recall')
            ax1.bar(x + width, f1_scores, width, label='F1-Score')
            ax1.set_xlabel('Camera')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics by Camera')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Camera {c}" for c in cameras])
            ax1.legend()
            ax1.set_ylim(0, 1.1)
            
            # Confusion matrix visualization
            for i, camera_id in enumerate(cameras):
                acc_data = accuracy['camera_wise_accuracy'][camera_id]
                conf_matrix = np.array([
                    [acc_data['true_positives'], acc_data['false_negatives']],
                    [acc_data['false_positives'], 0]  # True negatives not applicable here
                ])
                
                if i == 0:  # Show only first camera's confusion matrix
                    im = ax2.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
                    ax2.set_title(f'Detection Matrix - Camera {camera_id}')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Actual')
                    ax2.set_xticks([0, 1])
                    ax2.set_yticks([0, 1])
                    ax2.set_xticklabels(['Detected', 'Not Detected'])
                    ax2.set_yticklabels(['Present', 'Not Present'])
                    
                    # Add text annotations
                    for ii in range(2):
                        for jj in range(2):
                            ax2.text(jj, ii, conf_matrix[ii, jj], ha="center", va="center")
            
            # Global statistics
            ax3.bar(['Ground Truth', 'Predicted'], 
                   [accuracy['total_ground_truth_vehicles'], accuracy['total_predicted_global_ids']])
            ax3.set_title('Vehicle Count Comparison')
            ax3.set_ylabel('Number of Vehicles')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_evaluation.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create ground truth visualization
        self._create_ground_truth_visualization(output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _create_ground_truth_visualization(self, output_dir: str):
        """Create visualization of ground truth data"""
        if not self.ground_truth:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Ground Truth Data Analysis', fontsize=16)
        
        # Vehicle counts per camera
        cameras = list(self.ground_truth.keys())
        vehicle_counts = [len(tracks) for tracks in self.ground_truth.values()]
        
        ax1.bar([f"Camera {c}" for c in cameras], vehicle_counts)
        ax1.set_title('Ground Truth Vehicles per Camera')
        ax1.set_ylabel('Number of Vehicles')
        
        # Vehicle track lengths (if available)
        all_track_lengths = []
        for camera_id, tracks in self.ground_truth.items():
            for vehicle_id, track in tracks.items():
                if isinstance(track, list):
                    all_track_lengths.append(len(track))
        
        if all_track_lengths:
            ax2.hist(all_track_lengths, bins=20, alpha=0.7)
            ax2.set_title('Vehicle Track Length Distribution')
            ax2.set_xlabel('Track Length (frames)')
            ax2.set_ylabel('Number of Vehicles')
        else:
            ax2.text(0.5, 0.5, 'No track length data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Vehicle Track Length Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ground_truth_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = './analysis_output/reid_analysis_report.txt'):
        """Generate a comprehensive text report"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("Vehicle Re-Identification System Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated at: {datetime.now()}\n\n")
            
            # Detection statistics
            stats = self.analyze_detection_statistics()
            f.write("1. DETECTION STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total detections: {stats['total_detections']}\n")
            f.write(f"Unique local vehicle IDs: {stats['unique_vehicle_count']}\n")
            f.write(f"Global vehicle IDs: {stats['global_id_count']}\n\n")
            
            f.write("Detections per camera:\n")
            for camera_id, count in stats['detections_per_camera'].items():
                f.write(f"  Camera {camera_id}: {count}\n")
            
            f.write("\nVehicle type distribution:\n")
            for vehicle_type, count in stats['detections_per_type'].items():
                percentage = (count / stats['total_detections']) * 100
                f.write(f"  {vehicle_type}: {count} ({percentage:.1f}%)\n")
            
            # Cross-camera analysis
            cross_camera = self.analyze_cross_camera_performance()
            f.write("\n\n2. CROSS-CAMERA RE-IDENTIFICATION\n")
            f.write("-" * 35 + "\n")
            f.write(f"Cross-camera vehicles: {cross_camera['cross_camera_count']}\n")
            f.write(f"Single-camera vehicles: {cross_camera['single_camera_count']}\n")
            f.write(f"Cross-camera rate: {cross_camera['cross_camera_rate']:.3f}\n\n")
            
            if cross_camera['cross_camera_vehicles']:
                f.write("Top cross-camera matches:\n")
                sorted_matches = sorted(cross_camera['cross_camera_vehicles'], 
                                      key=lambda x: x['detections'], reverse=True)
                for i, match in enumerate(sorted_matches[:10]):
                    f.write(f"  {i+1}. {match['global_id']}: {match['detections']} detections "
                           f"across cameras {match['cameras']}\n")
            
            # Accuracy evaluation
            accuracy = self.evaluate_reid_accuracy()
            f.write("\n\n3. PERFORMANCE EVALUATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"Ground truth vehicles: {accuracy['total_ground_truth_vehicles']}\n")
            f.write(f"Predicted global IDs: {accuracy['total_predicted_global_ids']}\n\n")
            
            if accuracy['camera_wise_accuracy']:
                f.write("Camera-wise performance:\n")
                for camera_id, metrics in accuracy['camera_wise_accuracy'].items():
                    f.write(f"  Camera {camera_id}:\n")
                    f.write(f"    Precision: {metrics['precision']:.3f}\n")
                    f.write(f"    Recall: {metrics['recall']:.3f}\n")
                    f.write(f"    F1-Score: {metrics['f1_score']:.3f}\n")
            
            # System overview
            f.write("\n\n4. SYSTEM OVERVIEW\n")
            f.write("-" * 18 + "\n")
            f.write(f"Cameras analyzed: {len(CAMERA_CONFIG)}\n")
            for camera_key, camera_config in CAMERA_CONFIG.items():
                f.write(f"  {camera_config['name']} (ID: {camera_config['id']})\n")
            
            f.write(f"\nResults file: {self.results_file}\n")
            f.write(f"ReID state directory: {self.reid_state_dir}\n")
        
        print(f"Analysis report saved to {output_file}")

def main():
    """Main function for running analysis"""
    print("Vehicle ReID System Analysis")
    print("=" * 30)
    
    # Create analyzer
    analyzer = VehicleReIDAnalyzer()
    
    # Check if we have results to analyze
    if not analyzer.reid_results:
        print("\nNote: No ReID results found. To get meaningful analysis:")
        print("1. First run the ReID producer: python producer.py")
        print("2. Then run the ReID consumer: python consumer.py")
        print("3. Finally run this analyzer again: python analyzer.py")
        print("\nGenerating analysis with ground truth data only...\n")
    
    # Run analysis
    print("Generating visualizations...")
    analyzer.create_visualizations()
    
    print("Generating report...")
    analyzer.generate_report()
    
    # Print summary statistics
    stats = analyzer.analyze_detection_statistics()
    cross_camera = analyzer.analyze_cross_camera_performance()
    
    print("\nSUMMARY:")
    if stats['total_detections'] > 0:
        print(f"Total detections processed: {stats['total_detections']}")
        print(f"Unique vehicles identified: {stats['global_id_count']}")
        print(f"Cross-camera matches: {cross_camera['cross_camera_count']}")
        print(f"Cross-camera success rate: {cross_camera['cross_camera_rate']:.1%}")
    else:
        print("No ReID results to analyze")
    
    # Print ground truth summary
    total_ground_truth = sum(len(tracks) for tracks in analyzer.ground_truth.values())
    print(f"Ground truth vehicles available: {total_ground_truth}")
    print(f"Cameras with annotations: {len(analyzer.ground_truth)}")
    
    print("\nAnalysis complete! Check the 'analysis_output' directory for detailed results.")
    
    if not analyzer.reid_results:
        print("\n💡 Tip: Run the full ReID pipeline first to get comprehensive analysis!")

if __name__ == "__main__":
    main()
