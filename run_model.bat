@REM conda activate yuting
@REM 將C:\Users\yuting\AppData\Local\torch_extensions\torch_extensions\Cache 刪掉

@REM TEST=True or False
set TEST=False

if %TEST%==True (
    set train_kimg=1
    set train_snap=1
    set recommand_kimg=1
    set recommand_snap=1
) else (
    set train_kimg=200
    set train_snap=40
    set recommand_kimg=10
    set recommand_snap=5
)

@REM train
set model_dataset_name="ANR_ASF16-32/07"
python train.py --is_recommand False ^
--kimg %train_kimg% ^
--resume ANR_ASF.pkl ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--outdir=training_run/%model_dataset_name% ^
--batch 16 ^
--input_param_dims 19 ^
--snap %train_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM recommend
set model_dataset_name="ANR_ASF16-32/07"
python train.py --is_recommand True ^
--kimg %recommand_kimg% ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--resume training_run/%model_dataset_name%/Model.pkl ^
--input_param_dims 19 ^
--outdir=target_run/test ^
--batch 16 ^
--snap %recommand_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM train
set model_dataset_name="ANR_ASF16-32/09"
python train.py --is_recommand False ^
--kimg %train_kimg% ^
--resume ANR_ASF.pkl ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--outdir=training_run/%model_dataset_name% ^
--batch 16 ^
--input_param_dims 19 ^
--snap %train_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM recommend
set model_dataset_name="ANR_ASF16-32/09"
python train.py --is_recommand True ^
--kimg %recommand_kimg% ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--resume training_run/%model_dataset_name%/Model.pkl ^
--input_param_dims 19 ^
--outdir=target_run/%model_dataset_name% ^
--batch 16 ^
--snap %recommand_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 


@REM train
set model_dataset_name="ANR_ASF16-32/62"
python train.py --is_recommand False ^
--kimg %train_kimg% ^
--resume ANR_ASF.pkl ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--outdir=training_run/%model_dataset_name% ^
--batch 16 ^
--input_param_dims 19 ^
--snap %train_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM recommend
set model_dataset_name="ANR_ASF16-32/62"
python train.py --is_recommand True ^
--kimg %recommand_kimg% ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--resume training_run/%model_dataset_name%/Model.pkl ^
--input_param_dims 19 ^
--outdir=target_run/%model_dataset_name% ^
--batch 16 ^
--snap %recommand_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM train
set model_dataset_name="ANR_ASF16-32/67"
python train.py --is_recommand False ^
--kimg %train_kimg% ^
--resume ANR_ASF.pkl ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--outdir=training_run/%model_dataset_name% ^
--batch 16 ^
--input_param_dims 19 ^
--snap %train_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 

@REM recommend
set model_dataset_name="ANR_ASF16-32/67"
python train.py --is_recommand True ^
--kimg %recommand_kimg% ^
--dataset_paths=datasets/train/%model_dataset_name% ^
--resume training_run/%model_dataset_name%/Model.pkl ^
--input_param_dims 19 ^
--outdir=target_run/%model_dataset_name% ^
--batch 16 ^
--snap %recommand_snap% ^
--resolution 256 --lr 1e-3 --gpus 1 --gamma 10 --aug noaug --metrics True --eval_img_data None 
