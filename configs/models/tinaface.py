from collections import OrderedDict

cfg = dict(
    type="Tinaface",
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        norm_eval=False,
        dcn=dict(type="DCN", deform_groups=1, im2col_step=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        style="pytorch",
    ),
    neck=dict(
        type="Sequential",
        modules=[
            OrderedDict(
                fpn=dict(
                    type="FPN",
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    start_level=0,
                    add_extra_convs="on_input",
                    num_outs=6,
                    norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                    upsample_cfg=dict(mode="bilinear"),
                ),
                inception=dict(
                    type="Inception",
                    in_channels=256,
                    num_levels=6,
                    norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                    share=True,
                ),
            )
        ],
    ),
    bbox_head=dict(
        type="IoUAwareRetinaHead",
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8, 16, 32],
            ratios=[1.3],
            strides=[4, 8, 16, 32, 64, 128],
        ),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
    ),
    frame_to_result=dict(
        type="Frames2Results",
        frame_meta=dict(
            filename="",
            ori_filename="",
            ori_shape=(640, 480, 3),
            img_shape=(640, 480, 3),
            pad_shape=(640, 480, 3),
            flip=False,
            flip_direction="horizontal",
            scale_factor=[1.0, 1.0, 1.0, 1.0],
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[1.0, 1.0, 1.0],
                to_rgb=True,
            ),
        ),
        meshgrid=dict(
            type="BBoxAnchorMeshGrid",
            strides=[4, 8, 16, 32, 64, 128],
            base_anchor=dict(
                octave_base_scale=2 ** (4 / 3),
                scales_per_octave=3,
                ratios=[1.3],
                base_sizes=[4, 8, 16, 32, 64, 128],
            ),
        ),
        converter=dict(
            type="BBoxAnchorConverter",
            num_classes=1,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            nms_pre=10000,
            use_sigmoid=True,
        ),
        score_thr=0.4,
        nms_cfg=dict(type="nms", iou_threshold=0.45),
        max_num=1,
        return_inds=False,
    ),
    extract_bboxes=dict(type="ExtractBBoxes", single=True),
    init_cfg=dict(type="PretrainedInit", checkpoint="weights/tinaface_r50_fpn_bn.pth"),
)
