from tennis.dataset import TennisDataset, TennisPoseDataset
from tennis.baseline.model import TennisClassifier
from tennis.attention.model import AttentionTennisClassifier
from tennis.skeleton.model import SkeletonTennisClassifier
from tennis.skeleton_attention.model import SkeletonAttentionTennisClassifier

MODEL_CONFIG = {
    "baseline": {
        "class": TennisClassifier,
        "dataset": TennisDataset,
        "requires_pose": False,
    },
    "attention": {
        "class": AttentionTennisClassifier,
        "dataset": TennisDataset,
        "requires_pose": False,
    },
    "skeleton": {
        "class": SkeletonTennisClassifier,
        "dataset": TennisPoseDataset,
        "requires_pose": True,
    },
    "skeleton_attention": {
        "class": SkeletonAttentionTennisClassifier,
        "dataset": TennisPoseDataset,
        "requires_pose": True,
    },
}

def get_model_class(model_type):
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIG[model_type]["class"]

def get_dataset_class(model_type):
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIG[model_type]["dataset"]

def requires_pose(model_type):
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIG[model_type]["requires_pose"]

def get_model_choices():
    return list(MODEL_CONFIG.keys())
