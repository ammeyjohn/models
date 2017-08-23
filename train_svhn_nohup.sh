nohup python -u object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=/home/ubuntu/workspace/dataset/model/rfcn_resnet101_svhn.config \
	--train_dir=/home/ubuntu/workspace/output 1>train.log 2>&1 &
