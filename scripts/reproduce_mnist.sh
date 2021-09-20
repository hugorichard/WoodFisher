for target in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python ./main.py --exp_name=mplnet-mnist --dset=mnist --dset_path=../datasets --arch=mlpnet --config_path=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id 64 --fisher-damp 1e-5 --prune-modules fc1_fc2_fc3 --fisher-subsample-size 100 --fisher-mini-bsz 1 --update-config --prune-class woodfisherblock --target-sparsity $target --prune-end 25 --prune-freq 25 --seed 1 --deterministic --full-subsample --one-shot --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt --not-oldfashioned --use-model-config --woodburry-joint-sparsify --result-file result/mlpnet-mnist
done
