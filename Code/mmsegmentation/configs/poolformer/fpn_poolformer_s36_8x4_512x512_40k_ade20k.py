_base_ = './fpn_poolformer_s12_8x4_512x512_40k_ade20k.py'
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        arch='s36',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')))
