python train_image_classifier.py \
  --train_dir=./train\
  --dataset_dir=./data \
  --dataset_name=meters \
  --dataset_split_name=train \
  --model_name=inception_v3 \
  --checkpoint_path=./model/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --clone_on_cpu=True
