"""
Simple video-language dataset class file, for illustration purposes. See comments below for relevant highlights.
"""

import os
import pandas as pd
import torch
from torch.utils import data

class VideoLanguageDataset(data.Dataset):
    """
    Example (simple) video-language dataset class, for illustration purposes for ATP training.
    """
    def __init__(self, args, split="train", **kwargs):
        super().__init__()
        self.data_path = args.data_path
        self.split = split
        self.get_text_query = args.use_text_query
        self.get_text_cands = args.use_text_cands
        self.n_frames = args.n_frames
        
        # assuming a metadata file with a list of feature locations, etc.
        # edit to fit your dataset!
        self.metadata = pd.read_csv(os.path.join(self.data_path, f"metadata_{split}.csv"), index_col=0)
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, index):
        """
        Assuming torch files for each of the features for simplicity; adjust to fit your extracted features.
        (e.g. numpy, hdf5, json, etc.)
        """
        info_i = self.metadata.iloc[index]
        
        # get random frame sample, without replacement
        video_features = torch.load(info_i["video_features"])  # (L_video, D_in); L_video >> L
        frame_idxs_gt = torch.randperm(len(video_features))[:self.n_frames]
        video_features_sampled = video_features[frame_idxs_gt]  # (L, D_in)
        
        # get other features / labels
        text_query_features = torch.load(info_i["text_query_features"]) if self.get_text_query else []
        text_cands_features = torch.load(info_i["text_cands_features"]) if self.get_text_cands else []
        labels_gt = torch.load(info_i["labels_gt"])
        
        return video_features_sampled, frame_idxs_gt, text_query_features, text_cands_features, labels_gt
        
        
        
        
        

