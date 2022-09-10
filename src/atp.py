"""
Main file containing core ATP class code and some example utilities using ATP.
"""

from atp_utils import ModalityEmbeddings
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


@dataclass
class ATPConfig:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder).
    '''
    # ATPEncoder params
    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 128
    d_model_ff: int = 128
    enc_dropout: float = 0.1
    use_text_query: bool = False  # at least one use_text_* needs to be true for ATP to be multimodal
    use_text_cands: bool = False  # ^ see above. (note: if both are false, ATP is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 512  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)
    
    @classmethod
    def from_args(cls, args):
        return cls(n_layers = args.n_layers,
                   n_heads = args.n_heads,
                   d_model = args.d_model,
                   d_model_ff = args.d_model_ff,
                   enc_dropout = args.enc_dropout,
                   use_text_query = args.use_text_query,
                   use_text_cands = args.use_text_cands,
                   n_cands = args.n_cands,
                   use_ste = args.use_ste,
                   sel_dropout = args.sel_dropout,
                   d_input = args.d_input)


class ATPEncoder(nn.Module):
    """
    The multimodal transformer encoder for the ATP model. For analysis purposes, the ATP encoder
    does not use any positional information (no positional encodings + transformer / self-attention)
    and is generally kept low-capacity. If the goal is raw accuracy (not analysis), you can relax these constraints.
    """
    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig with parameters for the (transformer-based, atemporal) encoder for ATP.
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.d_model = config.d_model
        self.dropout = nn.Dropout(p=config.enc_dropout)
        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)
        
        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor):
        """
        x_inputs: torch.tensor of shape (L, N, D)
        """
        L, N, D = x_inputs.size()  # (sequence_length, batch_size, d_model)
        assert D == self.d_model, "inputs dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        x_encoded += self.modality_encoding(x_encoded)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)
        return x_encoded


class ATPSelectorModel(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language 
    encoding and outputs a (discrete) selection over the input frames, to help analyze 
    downstream discriminative video-language tasks.
    """
    
    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(config, **kwargs)
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_query: Optional[torch.tensor] = None,
                x_txt_cands: Optional[torch.tensor] = None,
                **kwargs):
        """
        Performs the ATP selection operation on the input embeddings.
        Returns selected (unmodified) visual embeddings and selection mask.
        x_vis_seq: torch.tensor of shape (N, L, D_in) with visual embeddings of size D_in
        x_txt_query: torch.tensor of shape (N, D_in) with optional query text embeddings
        x_txt_cands: torch.tensor of shape (N, L_cands, D_in) with optional add'l text embeddings
        (optional) temperature: used when config.use_ste is set to False; set as keyword argument. Default = 1.0.
        """        
        N, L, D = x_vis_seq.size()  # (batch_size, sequence_length, feature_dimension)
        x_vis_seq = x_vis_seq.permute(1, 0, 2)    # make (L, N, D); sequence first
    
        n_text_feats = self.atp_encoder.modality_encoding.n_text_feats
        
        # combine inputs into one multimodal sequence
        x_inputs = []
        if self.config.use_text_query:
            assert x_txt_query is not None, "missing x_txt_query."
            x_inputs.append(x_txt_query.unsqueeze(0))
        if self.config.use_text_cands:
            assert x_txt_cands is not None, "missing x_txt_cands."
            x_inputs.append(x_txt_cands.permute(1,0,2))
        x_inputs.append(x_vis_seq)
        x_inputs = torch.cat(x_inputs, dim=0)
        
        # embed the input sequence to the (smaller) model dimension (d_model) with modality encodings.
        x_encoded = self.embedding(self.dropout(x_inputs))
        x_atp_encoded = self.atp_encoder(x_encoded)[n_text_feats:]
        
        # obtain selection scores (logits)
        x_logits = self.logits(self.dropout(x_atp_encoded))

        # get selection mask over the visual inputs (frames)
        if self.training:
            # obtain differentiable selection mask during training.
            if self.config.use_ste:  # gumbel softmax straight-through estimator; hard selection
                selection_mask = F.gumbel_softmax(x_logits, dim=0, hard=True)
            else:  # softmax with temperature; soft selection
                selection_mask = F.softmax(x_logits / kwargs.get("temperature", 1.0), dim=0)
        else:
            # ATP always performs hard (discrete) selection during evaluation.
            selection_index_argmax = x_logits.max(dim=0, keepdim=True)[1]
            selection_mask = torch.zeros_like(x_logits, memory_format=torch.contiguous_format).scatter_(
                dim=0, index=selection_index_argmax, value=1.0)

        # use mask to perform selection
        selected_frames = (selection_mask * x_vis_seq).sum(dim=0)
        
        ret = [selected_frames, selection_mask]
        if not self.training: # extra logging during validation
            ret.append(x_logits)
        return tuple(ret)


######
# Below are some utility functions (and illustrative examples) for using ATP in the context of a 
# broader script, for visualization and inference. See also training_example.py.
def get_selected_index_from_selection_mask(frame_idxs_gt, selection_mask, sequence_first=False):
    """
    Quick utility helper method to get the "groundtruth" frame index selected
    (assuming shuffled input, and groundtruth frame indexes of (N, L) are available).
    This is useful for visualizations of ATP predictions on the original (ordered) video.
    """
    fidxs_gt = frame_idxs_gt.transpose(0, 1) if not sequence_first else frame_idxs_gt  # (L, N)
    return (selection_mask.squeeze(-1) * fidxs_gt).sum(dim=0)


def atp_downstream_task_forward(atp_selector: ATPSelectorModel, batch, **kwargs):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """
    x_vis_seq, frame_idxs_gt, x_txt_query, x_txt_cands, y_gt = batch
    # note: frame_idxs_gt only here for potential visualization; not used by ATP.
    selected_frames, *out_masks = atp_selector(x_vis_seq, x_txt_query, x_txt_cands, **kwargs)
    y_pred = F.cosine_similarity(selected_frames.unsqueeze(1), x_txt_cands, dim=-1)  # (N, N_ans)
    loss = F.cross_entropy(y_pred, y_gt)
    accs = (y_pred.argmax(dim=-1) == y_gt).float()
    return (loss, accs, selected_frames, y_pred, out_masks)

def apply_additional_masks(x, additional_masks=[]):
    """
    To enable combination of ATP with other (complementary) techniques, we provide this example (optional) function.
    Use to combine the outputs (masks) of these other methods with ATP. Does nothing if additional_masks is empty.
    Modify to fit your specific task use case.
    """
    x_out = x
    for mask in additional_masks:
        x_out *= mask
    return x_out

def atp_downstream_task_forward_with_additional_masks(atp_selector: ATPSelectorModel, batch, **kwargs):
    """
    Replica of atp_downstream_task_forward, with some examples for incorporating additional masks.
    Note: default behavior of this function is identical to atp_downstream_task_forward without 
    any additional masks. Modify to fit your specific task use case.
    """
    x_vis_seq, frame_idxs_gt, x_txt_query, x_txt_cands, y_gt, *additional_masks = batch
    # note: frame_idxs_gt only here for potential visualization; not used by ATP.
    if kwargs.get("apply_additional_masks_pre_atp", False):
        assert len(additional_masks) > 0, "additional_masks is empty, nothing to apply pre-ATP"
        x_txt_cands = apply_additional_masks(x_txt_cands, additional_masks)
    selected_frames, *out_masks = atp_selector(x_vis_seq, x_txt_query, x_txt_cands, **kwargs)

    y_pred = F.cosine_similarity(selected_frames.unsqueeze(1), x_txt_cands, dim=-1)  # (N, N_ans)
    if kwargs.get("apply_additional_masks_preds", False):
        assert len(additional_masks) > 0, "additional_masks is empty, nothing to apply post-ATP"
        y_pred = apply_additional_masks(y_pred, additional_masks)
    
    loss = F.cross_entropy(y_pred, y_gt)
    accs = (y_pred.argmax(dim=-1) == y_gt).float()
    return (loss, accs, selected_frames, y_pred, out_masks)

######