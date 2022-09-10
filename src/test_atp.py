"""
Simple test script to see if ATP dependencies are set-up correctly.
"""

import atp
import torch

if __name__ == '__main__':
    print("creating ATPConfig...")
    config = atp.ATPConfig(use_text_query=True, use_text_cands=True, use_ste=True)
    print("creating torch tensor inputs...")
    N, L, C, D = 5, 8, config.n_cands, config.d_input
    x_vis_seq = torch.rand(N, L, D)
    x_txt_query = torch.rand(N, D)
    x_txt_cands = torch.rand(N, C, D)
    print("creating ATPSelectorModel from config...")
    atp_model = atp.ATPSelectorModel(config)
    atp_model.train()
    print("running forward pass on random inputs...")
    out = atp_model(x_vis_seq, x_txt_query, x_txt_cands)
    print("dumping portion of random output from forward pass:")
    print(out[0])
    print("** success! (note: this script doesn't test cuda/gpu)")
