@REM python train.py --is_recommand False --input_param_dim 7 --lr 1e-6 --outdir=training_run --run_dir c7_region3_from_region2_pkl_lr_1e-6 --img_data=datasets/gain3_train --gpus 1 --kimg 100 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "network-snapshot-000077.pkl" --snap 1 


@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_origin_gain2_pkl_lr_1e-3_14k_img --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\3-gain3_train gain2.pkl lr1e-3\network-snapshot-000014.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_origin_gain2_pkl_lr_1e-3_42k_img --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\3-gain3_train gain2.pkl lr1e-3\network-snapshot-000042.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_origin_gain2_pkl_lr_1e-3_88k_img --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\3-gain3_train gain2.pkl lr1e-3\network-snapshot-000088.pkl" --snap 1 

@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_origin_gain2_pkl_lr_1e-6_20k_img --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region3_from_region2_pkl_lr_1e-6\network-snapshot-000016.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_origin_gain2_pkl_lr_1e-6_50k_img --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region3_from_region2_pkl_lr_1e-6\network-snapshot-000044.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_origin_gain2_pkl_lr_1e-6_100k_img --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region3_from_region2_pkl_lr_1e-6\network-snapshot-000095.pkl" --snap 1 

@REM 直接用region2的pkl
@REM python train.py --is_recommand True --input_param_dim 7 --lr 1e-3 --outdir=target_run --run_dir c7_region3_from_region2_pkl_no_train --img_data=datasets/c7_gain3_origin --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "network-snapshot-000077.pkl" --snap 1

@REM map=5
@REM python train.py --is_recommand False --input_param_dim 12 --lr 1e-3 --outdir=training_run --run_dir c7_region0_from_places_pkl --img_data=datasets/region0_train_align --gpus 1 --kimg 50 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "places.pkl" --snap 1 
@REM python train.py --is_recommand False --input_param_dim 12 --lr 1e-3 --outdir=training_run --run_dir c7_region1_from_places_pkl --img_data=datasets/region1_train_align --gpus 1 --kimg 50 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "places.pkl" --snap 1 
@REM python train.py --is_recommand False --input_param_dim 12 --lr 1e-3 --outdir=training_run --run_dir c7_region2_from_places_pkl --img_data=datasets/region2_train_align --gpus 1 --kimg 50 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "places.pkl" --snap 1 

@REM python train.py --is_recommand False --input_param_dim 12 --lr 1e-6 --outdir=training_run --run_dir c7_region0_from_region2_35k_pkl_lr_1e-6 --img_data=datasets/region0_train_align --gpus 1 --kimg 50 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 
@REM python train.py --is_recommand False --input_param_dim 12 --lr 1e-6 --outdir=training_run --run_dir c7_region1_from_region2_35k_pkl_lr_1e-6 --img_data=datasets/region1_train_align --gpus 1 --kimg 50 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 
@REM python train.py --is_recommand False --input_param_dim 12 --lr 1e-6 --outdir=training_run --run_dir c7_region2_from_region2_35k_pkl_lr_1e-6 --img_data=datasets/region2_train_align --gpus 1 --kimg 50 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 

@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region0_use_region0_from_places_pkl_35K --img_data=datasets/region0_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region0_from_places_pkl\network-snapshot-000029.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region1_use_region1_from_places_pkl_35K --img_data=datasets/region1_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region1_from_places_pkl\network-snapshot-000035.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region2_use_region2_from_places_pkl_35K --img_data=datasets/region2_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 

@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region0_use_region2_from_places_pkl_35K --img_data=datasets/region0_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region1_use_region2_from_places_pkl_35K --img_data=datasets/region1_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region2_use_region2_from_places_pkl_35K --img_data=datasets/region2_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region2_from_places_pkl\network-snapshot-000035.pkl" --snap 1 
python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region2_use_region1_from_places_pkl_35K --img_data=datasets/region2_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region1_from_places_pkl\network-snapshot-000035.pkl" --snap 1 


@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region0_use_region0_from_region2_pkl_35k_lr_1e-6_17K --img_data=datasets/region0_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region0_from_region2_35k_pkl_lr_1e-6\network-snapshot-000017.pkl" --snap 1 
@REM python train.py --is_recommand True --input_param_dim 12 --lr 1e-3 --outdir=target_run --run_dir c7_region1_use_region1_from_region2_pkl_35k_lr_1e-6_16K --img_data=datasets/region1_target_align --gpus 1 --kimg 3 --gamma 10 --aug noaug --metrics True --eval_img_data None --batch 2 --resume "training_run\c7_region1_from_region2_35k_pkl_lr_1e-6\network-snapshot-000016.pkl" --snap 1 