for ds in 32; do
  echo "ds = $ds, state_stride = 2"
  python train.py \
    --device cuda \
    --model gpt2_state \
    --gpt2_name openai-community/gpt2 --local_files_only \
    --inject_layer 8 \
    --d_state $ds \
    --state_stride 2 \
    --schedule 64 --steps_per_stage 2000 \
    --eval_every 500 --log_every 100 \
    --eval_lens 512 \
    --eval_multi \
    --out_dir out_sweep_dstate_${ds}
done
