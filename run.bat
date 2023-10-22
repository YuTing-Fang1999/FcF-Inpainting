@REM Train Model
python train.py --is_recommand False --kimg 50 --resume places.pkl --img_data=datasets/region0 --outdir=training_run/region0 --batch 16 --input_param_dim 12 --lr 1e-3 --snap 1 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM Recommand Model
python train.py --is_recommand True --kimg 3 --resume places.pkl --img_data=datasets/region0 --outdir=target_run/region0 --batch 16 --input_param_dim 12 --lr 1e-3 --snap 1 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 