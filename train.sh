for ds in 2; do
  echo "ds = $ds"
  python train.py \
    --device cuda --model gpt2_state \
    --gpt2_name openai-community/gpt2 --local_files_only \
    --inject_layer 8 --d_state $ds \
    --schedule 64 --steps_per_stage 1000 \
    --eval_every 500 --log_every 100 \
    --eval_lens 512 \
    --eval_multi \
    --out_dir out_sweep_dstate_${ds}
done
