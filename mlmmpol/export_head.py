"""Export the trained head (qm_ref/heads/<version>/head.pt) as the plugin TorchScript module for one ligand.

alpha = AMOEBA (Poltype) polarizability × 1.10 of the ML atoms, in ML-region order; for a head trained with
learn_alpha it is the base of the correction α = α_base · exp(δα(R)).
Python usage: export_head('qm_ref/heads/v1_f128_b4_s0', z, charge, alpha_nm3, 'ben_head.pt')
"""
import json
from pathlib import Path

import torch

from .embedding_head import BoundHead, make_head

ALPHA_SCALE = 1.10


def load_head(path):
    path = Path(path)
    args = json.load(open(path / 'metrics.json'))['args']
    head = make_head(args.get('version', 1), n_features=args['features'], n_blocks=args['blocks'],
                     n_vec=args.get('n_vec', 4), learn_alpha=bool(args.get('learn_alpha', False))).double()
    state = torch.load(path / 'head.pt', map_location='cpu')
    missing, unexpected = head.load_state_dict(state, strict=False)
    # heads trained before α(R) have no alpha_out (left at zero: α = α_base)
    assert not unexpected and all(k.startswith('alpha_out.') for k in missing), (missing, unexpected)
    assert head.learn_alpha is False or not missing, missing
    head.eval()
    return head


def export_head(path, z, charge, alpha_amoeba_nm3, out, alpha_scale=ALPHA_SCALE):
    bound = BoundHead(load_head(path), [int(v) for v in z], float(charge), [a * alpha_scale for a in alpha_amoeba_nm3])
    scripted = torch.jit.script(bound)
    scripted.save(str(out))
    return scripted
