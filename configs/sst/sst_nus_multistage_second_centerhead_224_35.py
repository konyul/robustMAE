_base_ = [
    '../_base_/models/sst_base_nus_centerhead.py',
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py',
]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

voxel_size = (0.1, 0.1, 8)
window_shape=(12, 12) # 12 * 0.32m
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]



drop_info_training ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':144, 'drop_range':(100, 100000)},
}
drop_info_test ={
    0:{'max_tokens':30, 'drop_range':(0, 30)},
    1:{'max_tokens':60, 'drop_range':(30, 60)},
    2:{'max_tokens':100, 'drop_range':(60, 100)},
    3:{'max_tokens':144, 'drop_range':(100, 100000)},
}
drop_info = (drop_info_training, drop_info_test)
shifts_list=[(0, 0), (window_shape[0]//2, window_shape[1]//2) ]

file_client_args = dict(backend='disk')
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))


model = dict(
    type='DynamicVoxelNet',
    centerpoint_head=True,
    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='SSTInputLayer',
        window_shape=window_shape,
        shifts_list=shifts_list,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
    ),

    backbone=dict(
        type='SSTMultiStageSecondv1',
        num_stage=3,
        shifts_list=shifts_list,
        spatial_shape=[[1,1024,1024],[1,512,512],],
        bev_shapes=[[1024,1024],[512,512],[256,256],],
        d_model=[64,128,128,],
        nhead=[8,8,8 ],
        num_blocks=[2,2,4,],
        dim_feedforward=[128,256,256 ] ,
        output_shape=[256, 256],
        conv_in_channels=128,
        conv_out_channels=[128, 256],
        layer_nums=[3, 5],
        layer_strides=[1, 2],
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        window_shape=window_shape,
        #checkpoint_blocks=[0,1,2]
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[0.5, 1],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(in_channels=sum([256,256]),
                   bbox_coder=dict(
                        voxel_size=voxel_size[:2],
                        pc_range=point_cloud_range[:2],
                        out_size_factor=8,)),
    train_cfg=dict(point_cloud_range=point_cloud_range,
                    voxel_size= voxel_size,
                    grid_size=[1024, 1024, 1],
                    out_size_factor=8,),
    test_cfg=dict(pc_range=point_cloud_range[:2],
                    voxel_size= voxel_size,
                    out_size_factor=8,)
)

# runner = dict(type='EpochBasedRunner', max_epochs=12)
# evaluation = dict(interval=12)



train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

# fp16 = dict(loss_scale=512.0)
evaluation = dict(interval=20, pipeline=eval_pipeline)