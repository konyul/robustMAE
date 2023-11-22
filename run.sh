# CONFIG=sst_waymoD5_1x_3class_8heads_v2
# bash tools/dist_train.sh configs/sst_refactor/$CONFIG.py 8 --work-dir ./work_dirs/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./work_dirs/$CONFIG/results evaluation.metric=waymo
./tools/dist_train.sh configs/mae_sst/robust_mae_m_sst_nus_singlestage_curv_07_ssl_dataset_wo_dbsampler_6x_1e-5.py 1