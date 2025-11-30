import os
import collections

CLASS_MAPPING = {
    "backhand": ["backhand", "backhand_slice", "backhand2hands"],
    "forehand": ["forehand_flat", "forehand_openstands", "forehand_slice"],
    "service": ["flat_service", "kick_service", "slice_service"],
    "bvolley": ["backhand_volley"],
    "fvolley": ["forehand_volley"],
    "smash": ["smash"],
}

# Invert mapping for easy lookup
subdir_to_class = {}
for cls, subdirs in CLASS_MAPPING.items():
    for subdir in subdirs:
        subdir_to_class[subdir] = cls

data_dir = "data/VIDEO_RGB"
player_counts = collections.defaultdict(lambda: collections.defaultdict(int))

for root, dirs, files in os.walk(data_dir):
    for f in files:
        if not f.endswith((".avi", ".mp4", ".mov")):
            continue
            
        # Get subdir name
        subdir = os.path.basename(root)
        if subdir not in subdir_to_class:
            continue
            
        target_class = subdir_to_class[subdir]
        
        # Parse player ID
        # Filename: p<id>_...
        try:
            pid_part = f.split('_')[0]
            if pid_part.startswith('p'):
                pid = pid_part
                player_counts[pid][target_class] += 1
        except:
            pass

# Analyze
all_classes = sorted(list(CLASS_MAPPING.keys()))
all_players = sorted(list(player_counts.keys()), key=lambda x: int(x[1:]))

print(f"Total Players: {len(all_players)}")
print(f"Classes: {all_classes}")

uniform = True
first_player_counts = None

for pid in all_players:
    counts = player_counts[pid]
    # Check if this player has same counts for all classes? 
    # Or if all players have same counts for a specific class?
    # The user asked: "Confirm whether or not every player appears in every class the same number of times."
    # This implies: Count(Player X, Class Y) == Constant for all X, Y? 
    # Or maybe Count(Player X, Class Y) == Count(Player Z, Class Y)?
    
    # Let's print the matrix if it's small, or summarize.
    
    row = [counts[c] for c in all_classes]
    if first_player_counts is None:
        first_player_counts = row
    
    if row != first_player_counts:
        uniform = False

if uniform:
    print(f"CONFIRMED: Every player has exactly the same distribution: {dict(zip(all_classes, first_player_counts))}")
else:
    print("NOT UNIFORM. Distribution varies.")
    # Print a few examples
    print("Example distributions:")
    for pid in all_players[:5]:
        print(f"{pid}: {dict(player_counts[pid])}")
    if len(all_players) > 5:
        print("...")
