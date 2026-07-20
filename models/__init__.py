"""Model architectures used in the DiffSA-EEG comparison.

SSDA_Modular       DiffSA-EEG. `ModularDiffE` is the discriminative network with four independently
                   switchable components; `ConditionalUNet` + `DDPM` are the diffusion branch.
models_EEGNet      EEGNet (Lawhern et al., 2018)
models_Deep4Net    Deep4Net / ConvNet (Schirrmeister et al., 2017)
models_ChronoNet   ChronoNet (Roy et al., 2019)
models_BDTCN       temporal convolutional network (Gemein et al., 2020)
models_EEGConformer EEG-Conformer (Song et al., 2023)
models_EEGDeformer  EEG-Deformer (Ding et al., 2024)
models_ATCNet       ATCNet (Altaheri et al., 2023)

Submodules are imported lazily by train.py so that a missing optional dependency in one baseline
does not prevent the others from running.
"""
