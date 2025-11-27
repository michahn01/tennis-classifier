import os
from tennis.dataset import TennisDataset

def verify_dataset_split():
    root_dir = 'data/VIDEO_RGB'
    
    # Check if directory exists
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found. Skipping verification.")
        return

    # Test Case 1: experts_only=True (Default)
    print("\n--- Testing experts_only=True ---")
    dataset_experts = TennisDataset(root_dir=root_dir, split='train', experts_only=True)
    print(f"Total samples (experts only): {len(dataset_experts)}")
    
    non_expert_count = 0
    expert_count = 0
    for path, label in dataset_experts.samples:
        filename = os.path.basename(path)
        try:
            player_id_str = filename.split('_')[0]
            if player_id_str.startswith('p'):
                player_id = int(player_id_str[1:])
                if 32 <= player_id <= 55:
                    expert_count += 1
                else:
                    non_expert_count += 1
        except ValueError:
            pass
            
    print(f"Expert samples: {expert_count}")
    print(f"Non-expert samples: {non_expert_count}")
    
    if non_expert_count == 0 and expert_count > 0:
        print("SUCCESS: experts_only=True worked correctly.")
    else:
        print("FAILURE: experts_only=True failed.")

    # Test Case 2: experts_only=False
    print("\n--- Testing experts_only=False ---")
    dataset_all = TennisDataset(root_dir=root_dir, split='train', experts_only=False)
    print(f"Total samples (all): {len(dataset_all)}")
    
    non_expert_count = 0
    expert_count = 0
    for path, label in dataset_all.samples:
        filename = os.path.basename(path)
        try:
            player_id_str = filename.split('_')[0]
            if player_id_str.startswith('p'):
                player_id = int(player_id_str[1:])
                if 32 <= player_id <= 55:
                    expert_count += 1
                else:
                    non_expert_count += 1
        except ValueError:
            pass
            
    print(f"Expert samples: {expert_count}")
    print(f"Non-expert samples: {non_expert_count}")
    
    if non_expert_count > 0 and expert_count > 0:
        print("SUCCESS: experts_only=False included non-experts.")
    else:
        print("FAILURE: experts_only=False failed (expected mixed samples).")

if __name__ == "__main__":
    verify_dataset_split()
