import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hrformer.hrt import HRT
from models.hrformer.hilo_hrt import HiLoHRT
from models.decoder import RegressionDecoder, HRFormerSkipDecoder, MultiHeadPhysicsDecoder


class HRFormerPhysicsPretrainer(nn.Module):
    """HRFormer + shared skip-decoder body + multiple lightweight physics heads."""

    def __init__(self, cfg, head_specs):
        super().__init__()
        if cfg["model"]["backbone"]["type"] == "HRT":
            self.backbone = HRT(
                in_channels=cfg["model"]["in_channels"],
                extra=cfg["model"]["backbone"]["extra"],
                norm_cfg=cfg["model"]["norm_cfg"],
            )

        elif cfg["model"]["backbone"]["type"] == "HiLoHRT":
            self.backbone = HiLoHRT(
                in_channels=cfg["model"]["in_channels"],
                extra=cfg["model"]["backbone"]["extra"],
                norm_cfg=cfg["model"]["norm_cfg"],
            )

        else:
            raise ValueError(f"Unsupported backbone type: {cfg['type']}")

        dec_cfg = cfg["model"]["decoder"]
        self.decoder = MultiHeadPhysicsDecoder(
            backbone_channels=dec_cfg.get("in_channels", 32),
            stem2_channels=dec_cfg.get("stem2_channels", 64),
            stem1_channels=dec_cfg.get("stem1_channels", 64),
            hidden_channels=dec_cfg.get("hidden_channels", 64),
            head_specs=head_specs,
        )
        self.branch_index = cfg["model"].get("branch_index", 0)

    def forward(self, x):
        feats, stem_feats = self.backbone(x, return_stem_features=True)
        feat = feats[self.branch_index]
        return self.decoder(
            feat,
            stem1=stem_feats["stem1"],
            stem2=stem_feats["stem2"],
        )


def load_physics_pretrained_for_downstream(model, ckpt_path, device="cpu", verbose=True):
    """Load backbone and shared decoder body into downstream HRFormerRadioMapRegressor.

    Physics heads are ignored. Downstream radio head remains randomly initialized.
    This assumes the downstream decoder is HRFormerSkipDecoder with `decoder.body`.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    filtered = {}
    skipped = []
    own = model.state_dict()
    for k, v in state.items():
        # Keep backbone and shared decoder body. Drop physics-specific heads.
        if k.startswith("decoder.heads"):
            skipped.append(k)
            continue
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    msg = model.load_state_dict(filtered, strict=False)
    if verbose:
        print(f"Loaded {len(filtered)} tensors from physics pretraining: {ckpt_path}")
        print(f"Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)} | Skipped: {len(skipped)}")
    return msg


### downstream model ###
class HRFormerRadioMapRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg["model"]["backbone"]["type"] == "HRT":
            self.backbone = HRT(
                in_channels=cfg["model"]["in_channels"],
                extra=cfg["model"]["backbone"]["extra"],
                norm_cfg=cfg["model"]["norm_cfg"],
            )

        elif cfg["model"]["backbone"]["type"] == "HiLoHRT":
            self.backbone = HiLoHRT(
                in_channels=cfg["model"]["in_channels"],
                extra=cfg["model"]["backbone"]["extra"],
                norm_cfg=cfg["model"]["norm_cfg"],
            )

        else:
            raise ValueError(f"Unsupported backbone type: {cfg['type']}")

        if cfg["model"]["decoder"]["type"] == "RegressionDecoder":
            self.decoder = RegressionDecoder(
                backbone_channels=cfg["model"]["decoder"]["in_channels"],
                out_channels=cfg["model"]["decoder"]["out_channels"],
            )
        elif cfg["model"]["decoder"]["type"] == "HRFormerSkipDecoder":
            self.decoder = HRFormerSkipDecoder(
                backbone_channels=cfg["model"]["decoder"]["in_channels"],
                out_channels=cfg["model"]["decoder"]["out_channels"],
            )
        
        self.branch_index = cfg["model"]["branch_index"]
        self.use_stem_skip = cfg["model"]["use_stem_skip"]

    def forward(self, x):
        if self.use_stem_skip:
            feats, stem_feats = self.backbone(x, return_stem_features=True)
            feat = feats[self.branch_index]

            out = self.decoder(
                feat,
                stem1=stem_feats["stem1"],
                stem2=stem_feats["stem2"],
            )
        else:
            feats = self.backbone(x)
            feat = feats[self.branch_index]
            out = self.decoder(feat)

        return out