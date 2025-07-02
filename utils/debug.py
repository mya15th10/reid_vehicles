

import sys
import torch
import numpy as np
import inspect

sys.path.append('.')

def debug_model_signature():
    """Check model forward signature to understand required parameters"""
    print("=" * 60)
    print("STEP 1: CHECKING MODEL SIGNATURE")
    print("=" * 60)
    
    try:
        from model import make_model
        from config import cfg
        cfg.merge_from_file('configs/custom_vehicle.yml')
        
        # Create model and check forward signature
        model = make_model(cfg, num_class=486)
        print('Model forward signature:')
        sig = inspect.signature(model.forward)
        print(f"  {sig}")
        
        # Check if model has specific requirements
        params = list(sig.parameters.keys())
        print(f"Required parameters: {params}")
        
        if 'view_label' in params:
            print("Model expects view_label parameter")
        else:
            print(" Model does NOT expect view_label parameter")
            
    except Exception as e:
        print(f" Error checking model signature: {e}")

def debug_dataloader_output():
    """Check what validation loader actually returns"""
    print("\n" + "=" * 60)
    print("STEP 2: CHECKING DATALOADER OUTPUT")
    print("=" * 60)
    
    try:
        from datasets.make_dataloader import make_dataloader
        from config import cfg
        cfg.merge_from_file('configs/custom_vehicle.yml')
        
        train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
        
        print(f"Dataloader info:")
        print(f"  num_query: {num_query}")
        print(f"  num_classes: {num_classes}")
        print(f"  camera_num: {camera_num}")
        print(f"  view_num: {view_num}")
        
        # Check what validation loader actually returns
        print("\nChecking validation batch structure:")
        for i, batch in enumerate(val_loader):
            print(f"Batch {i}:")
            print(f"  Number of items: {len(batch)}")
            print(f"  Item types: {[type(x).__name__ for x in batch]}")
            
            if len(batch) >= 5:
                img, vid, camid, camids, target_view = batch[:5]
                print(f"  img shape: {img.shape}")
                print(f"  vid type: {type(vid)}, sample: {vid[:5] if len(vid) > 5 else vid}")
                print(f"  camid type: {type(camid)}, sample: {camid[:5] if len(camid) > 5 else camid}")
                print(f"  camids shape: {camids.shape}")
                print(f"  target_view type: {type(target_view)}")
                
                if hasattr(target_view, 'shape'):
                    print(f"  target_view shape: {target_view.shape}")
                    print(f"  target_view sample: {target_view[:5] if len(target_view) > 5 else target_view}")
                else:
                    print(f"  target_view value: {target_view}")
                    
            # Only check first batch
            break
            
    except Exception as e:
        print(f"Error checking dataloader: {e}")

def debug_evaluation_function():
    """Test the evaluation function with dummy data"""
    print("\n" + "=" * 60)
    print("STEP 3: CHECKING EVALUATION FUNCTION")
    print("=" * 60)
    
    try:
        from utils.metrics import R1_mAP_eval
        
        # Test with dummy data similar to real scenario
        num_query = 100
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes')
        
        print(f"Testing evaluation with {num_query} queries...")
        
        # Create realistic dummy data
        # 200 total samples: 100 query + 100 gallery
        total_samples = 200
        feature_dim = 768
        
        feat = torch.randn(total_samples, feature_dim)
        
        # Create PIDs with some overlap (realistic scenario)
        # 50 unique vehicles, each appears in both query and gallery
        unique_vehicles = 50
        pids = np.array([i % unique_vehicles for i in range(total_samples)])
        
        # Create camera IDs: query from cam 2, gallery from cam 3
        camids = np.array([2] * num_query + [3] * (total_samples - num_query))
        
        print(f"  Feature shape: {feat.shape}")
        print(f"  PIDs range: {pids.min()} to {pids.max()}")
        print(f"  Unique PIDs: {len(np.unique(pids))}")
        print(f"  Camera IDs: {np.unique(camids)}")
        
        # Test evaluation
        evaluator.reset()
        evaluator.update((feat, pids, camids))
        
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        print(f"Test evaluation successful:")
        print(f"  mAP: {mAP:.3f}")
        print(f"  Rank-1: {cmc[0]:.3f}")
        print(f"  Rank-5: {cmc[4]:.3f}")
        
        if mAP > 0.5:
            print("Evaluation function works correctly")
        else:
            print(" Low mAP in test - may indicate evaluation issue")
            
    except Exception as e:
        print(f" Error in evaluation function: {e}")
        import traceback
        traceback.print_exc()

def debug_dataset_splits():
    """Check dataset splits and ID distribution"""
    print("\n" + "=" * 60)
    print("STEP 4: CHECKING DATASET SPLITS DETAILS")
    print("=" * 60)
    
    try:
        from datasets.custom_vehicle_dataset import CustomVehicleDataset
        dataset = CustomVehicleDataset(root='./data/raw/CustomVehicleDataset')
        
        # Analyze training data
        train_pids = [item[1] for item in dataset.train]
        train_cams = [item[2] for item in dataset.train]
        
        print("Training data analysis:")
        print(f"  Unique PIDs: {len(set(train_pids))}")
        print(f"  PID range: {min(train_pids)} to {max(train_pids)}")
        print(f"  Camera IDs: {set(train_cams)}")
        
        # Analyze query data
        query_pids = [item[1] for item in dataset.query]
        query_cams = [item[2] for item in dataset.query]
        
        print("Query data analysis:")
        print(f"  Unique PIDs: {len(set(query_pids))}")
        print(f"  PID range: {min(query_pids)} to {max(query_pids)}")
        print(f"  Camera IDs: {set(query_cams)}")
        
        # Analyze gallery data
        gallery_pids = [item[1] for item in dataset.gallery]
        gallery_cams = [item[2] for item in dataset.gallery]
        
        print("Gallery data analysis:")
        print(f"  Unique PIDs: {len(set(gallery_pids))}")
        print(f"  PID range: {min(gallery_pids)} to {max(gallery_pids)}")
        print(f"  Camera IDs: {set(gallery_cams)}")
        
        # Check for potential issues
        if set(query_cams) == set(gallery_cams):
            print(" WARNING: Query and Gallery have same camera IDs!")
            print("   This might cause evaluation issues in re-ID")
            
        # Check PID overlap
        query_set = set(query_pids)
        gallery_set = set(gallery_pids)
        overlap = query_set.intersection(gallery_set)
        
        print(f"Query-Gallery overlap: {len(overlap)}/{len(query_set)} = {len(overlap)/len(query_set)*100:.1f}%")
        
        if len(overlap) < len(query_set) * 0.5:
            print("Low overlap between query and gallery PIDs!")
            
    except Exception as e:
        print(f" Error checking dataset splits: {e}")

def main():
    """Run all debug steps"""
    print("DEBUGGING LOW VALIDATION ACCURACY")
    print("Current results: mAP 0.7%, Rank-1 0.3%")
    print("Expected: 90%+ (testing on training data)")
    
    debug_model_signature()
    debug_dataloader_output()
    debug_evaluation_function()
    debug_dataset_splits()
    
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print("Based on debug results above:")
    print("1. If model doesn't need view_label → remove it from model calls")
    print("2. If query/gallery have same camera IDs → this explains low accuracy")
    print("3. If evaluation function fails → fix metrics.py")
    print("4. If target_view is wrong type → fix data handling")

if __name__ == "__main__":
    main()