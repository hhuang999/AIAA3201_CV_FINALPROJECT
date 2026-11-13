"""Keep the validated Small-model attention path stable across environments."""
from importlib import import_module

MODULES = {
    'unidepth': [
        'unidepth.models.backbones.metadinov2.attention',
        'unidepth.models.backbones.metadinov2.block',
    ],
    'dav2': [
        'depth_anything_v2.dinov2_layers.attention',
        'depth_anything_v2.dinov2_layers.block',
    ],
    'vda': [
        'video_depth_anything.dinov2_layers.attention',
        'video_depth_anything.dinov2_layers.block',
        'video_depth_anything.motion_module.attention',
        'video_depth_anything.motion_module.motion_module',
    ],
}


def use_pytorch_attention(model_name):
    """Select upstream PyTorch fallbacks without editing pinned upstream files.

    The verified run used these branches because the optional xformers import
    failed in that environment. A clean install must make the same choice.
    Small checkpoints use MLP feed-forward blocks, so optional SwiGLU classes
    are outside this inference path.
    """
    for name in MODULES[model_name]:
        module = import_module(name)
        if not hasattr(module,'XFORMERS_AVAILABLE'):
            raise ValueError(f'Pinned upstream attention selector is absent in {name}')
        module.XFORMERS_AVAILABLE = False
    return {'implementation':'upstream PyTorch attention branches',
            'xformers_attention_enabled':False,'configured_modules':MODULES[model_name]}
