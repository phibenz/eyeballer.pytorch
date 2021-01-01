python3 train_model.py \
  --dataset websites --arch vgg16 --seed 111 \
  --batch-size 32 --learning-rate 0.001 \
  --epochs 30 --schedule 12 25 --gammas 0.1 0.1 --workers 4

python3 train_model.py \
  --dataset websites --arch vgg19 --seed 111 \
  --batch-size 32 --learning-rate 0.001 \
  --epochs 30 --schedule 12 25 --gammas 0.1 0.1 --workers 4

python3 train_model.py \
  --dataset websites --arch resnet18 --seed 111 \
  --batch-size 32 --learning-rate 0.001 \
  --epochs 30 --schedule 12 25 --gammas 0.1 0.1 --workers 4

python3 train_model.py \
  --dataset websites --arch resnet50 --seed 111 \
  --batch-size 32 --learning-rate 0.001 \
  --epochs 30 --schedule 12 25 --gammas 0.1 0.1 --workers 4

python3 train_model.py \
  --dataset websites --arch resnet152 --seed 111 \
  --batch-size 32 --learning-rate 0.001 \
  --epochs 30 --schedule 12 25 --gammas 0.1 0.1 --workers 4