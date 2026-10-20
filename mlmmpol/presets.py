"""Presets of the 'ml' mode (head + coupling) in qm_ref/heads/presets.json; 'default' = the one chosen by validation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / 'qm_ref' / 'heads' / 'presets.json'


def ml_preset(name=None, root=None):
    """Return dict(name, head (path), a_cross, thole_ml, note) of the preset (default: 'default')."""
    base = Path(root) if root else ROOT
    cfg = json.loads((base / 'qm_ref' / 'heads' / 'presets.json').read_text())
    name = name or cfg['default']
    p = dict(cfg['presets'][name])
    p['name'] = name
    p['head'] = str(base / 'qm_ref' / 'heads' / p['head'])
    return p
