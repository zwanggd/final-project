default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=True,
        show=False,
        draw_gt=True,
        draw_pred=True),
)

# Logging
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

load_from = None
resume = False

# Evaluation
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='TensorboardVisBackend', save_dir='/home/zw4603/GSR/mmocr/tensorboard_logs')]
visualizer = dict(
    type='TextDetLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends)
