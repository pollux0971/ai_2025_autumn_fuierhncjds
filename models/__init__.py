# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from .DN_DAB_DETR import build_DABDETR

# Try to import deformable models, but don't fail if CUDA operators aren't compiled
try:
    from .dn_dab_deformable_detr import build_dab_deformable_detr
    from .dn_dab_deformable_detr_deformable_encoder_only import build_dab_deformable_detr_deformable_encoder_only
    from .dn_dab_dino_deformable_detr import build_dab_dino_deformable_detr
    DEFORMABLE_AVAILABLE = True
    print("Deformable models loaded successfully.")
except ImportError as e:
    print(f"Warning: Deformable models not available ({e}). Only DN-DAB-DETR will work.")
    # Define dummy functions to avoid import errors
    def build_dab_deformable_detr(*args, **kwargs):
        raise RuntimeError("Deformable models not available. Please compile CUDA operators first.")
    def build_dab_deformable_detr_deformable_encoder_only(*args, **kwargs):
        raise RuntimeError("Deformable models not available. Please compile CUDA operators first.")
    def build_dab_dino_deformable_detr(*args, **kwargs):
        raise RuntimeError("Deformable models not available. Please compile CUDA operators first.")
    DEFORMABLE_AVAILABLE = False
