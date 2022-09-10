"""
Supporting file for atp.py.
"""

from enum import IntEnum
import torch
from torch import nn

class ModalityEmbeddingsID(IntEnum):
    TEXT_QUESTION = 0
    TEXT_EMBEDDING = 1
    TEXT_UNUSED = 2  # ignore
    VISUAL_EMBEDDING = 3
    VISUAL_UNUSED = 4  # ignore

class ModalityEmbeddings(nn.Module):
    """
    Provides embeddings that indicate type of modality; for use with multimodal inputs for ATP. See atp.py for usage.
    """
    def __init__(self,
                 d_model: int,
                 use_text_query: bool = False,
                 use_text_cands: bool = False,
                 n_cands: int = 5):
        """
        Details for each of these arguments are provided in ATPConfig.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

        self.use_text_query = use_text_query
        self.use_text_cands = use_text_cands
        self.n_cands = n_cands if use_text_cands else 0
        self.n_text_feats = 1 if use_text_query else 0
        if use_text_cands:
            self.n_text_feats += n_cands

    def forward(self, x: torch.tensor):
        """
        x: torch.tensor of size (L, N, D)
        returns modality embeddings for x of size (L, *, D)
        """
        L, N, D = x.size()  # (sequence_length, batch_size, feature_dim)
        n_frames = L - self.n_text_feats
        
        # assemble the IDs for the modality encodings, language inputs then vision inputs
        class_ids = []
        if self.use_text_query:
            class_ids = [ModalityEmbeddingsID.TEXT_QUESTION,]
        if self.use_text_cands:
            class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING,] * self.n_cands)
        class_ids.extend([ModalityEmbeddingsID.VISUAL_EMBEDDING,] * n_frames)
        
        class_ids = torch.tensor(
            class_ids,
            dtype=torch.long,
            device=x.device
        ).unsqueeze(-1)
        
        # return modality embeddings
        return self.embedding(class_ids)
