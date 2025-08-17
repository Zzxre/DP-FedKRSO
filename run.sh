i=0
gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
rank=4 # set to rank=0 for full fine-tuning
kseed=10 # only work when agg
for alpha in -1; do # -1=iid, 0.5, 0.25
  for task in "qnli"; do #   "qnli" "cola" "sst2" "stsb" "mnli_mismatched"; do
    for method in "rso" "normal" "ffa"; do
      for interval in 10; do
        for batchsize in 32; do
          for seed in 0 1 2; do
            for lr in 2e-4; do
               cuda_device=${gpus[$((i % num_gpus))]}
              echo "Running with seed=${seed} on CUDA device=${cuda_device}"
              echo "python fed_train_glue_RSO.py --task ${task} --device ${cuda_device} --batch_size ${batchsize} --seed $((seed)) --lr ${lr} --agg_type ${method} --interval ${interval} --amp --lora_r ${rank} --alpha ${alpha} --kseed ${kssed} --wandb"
              # 实际运行命令
              python fed_train_glue_RSO.py --task ${task} --device ${cuda_device} --batch_size ${batchsize} --seed $((seed)) --lr ${lr} --agg_type ${method} --interval ${interval} --amp --lora_r ${rank} --alpha ${alpha} --kseed ${kseed} --wandb&
              ((i+=1))
            done
          done
        done
      done
    done
  done
done


