import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from a5_core import generate_a5, RandomSeqFinalDataset, eval_final_acc
from models import (
    BaselineAdapter,
    GRUBaseline,
    A5ExactScan,
    Route1SoftScan,
    GPT2FrozenBaseline,
    GPT2FrozenStateFusion,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()

    # basic
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    # model selection
    p.add_argument(
        "--model",
        type=str,
        default="route1",
        choices=["adapter", "gru", "exact", "route1", "gpt2", "gpt2_state"],
    )
    p.add_argument("--d_model", type=int, default=128)

    # adapter
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])

    # gru
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--gru_dropout", type=float, default=0.0)

    # route1
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--aux_weight", type=float, default=5.0)
    p.add_argument("--anneal_aux", action="store_true")

    # gpt2 frozen / state plugin
    p.add_argument("--gpt2_name", type=str, default="openai-community/gpt2")
    p.add_argument("--inject_layer", type=int, default=8)
    p.add_argument("--d_state", type=int, default=128)
    p.add_argument("--state_stride", type=int, default=1,
                   help="For gpt2_state: refresh injected teacher state every K steps (K>=1).")
    p.add_argument("--local_files_only", action="store_true")

    # ---- TRAIN-TIME ablations (normally keep FALSE for clean training) ----
    # (You can still use these if you want to do "training-time ablation",
    #  but for Scheme B, keep them OFF and use eval-only ablations below.)
    p.add_argument("--shuffle_state", action="store_true")
    p.add_argument("--reset_state", action="store_true")
    p.add_argument("--gate_zero", action="store_true")

    # ---- EVAL-ONLY ablations (Scheme B: causal intervention at eval) ----
    # If enabled, eval will report multiple tags: clean + selected eval-only interventions.
    p.add_argument("--eval_multi", action="store_true",
                   help="If set, evaluate clean and selected eval-only interventions each eval event.")
    p.add_argument("--eval_gate_zero", action="store_true",
                   help="Eval-only: gate=0 for gpt2_state (state channel disabled).")
    p.add_argument("--eval_shuffle_state", action="store_true",
                   help="Eval-only: shuffle teacher states over time.")
    p.add_argument("--eval_reset_state", action="store_true",
                   help="Eval-only: reset teacher state each step.")

    # optimization
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # data
    p.add_argument("--train_samples", type=int, default=200000)
    p.add_argument("--test_samples_per_len", type=int, default=10000)
    p.add_argument("--schedule", type=str, default="64")
    p.add_argument("--steps_per_stage", type=int, default=5000)
    p.add_argument("--eval_lens", type=str, default="64,128,256,512")

    # logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="outputs")

    # mechanism ablations for exact/route1 (executor-side)
    p.add_argument("--no_scan", action="store_true")
    p.add_argument("--shuffle_M", action="store_true")
    p.add_argument("--reset_each_step", action="store_true")

    return p.parse_args()


def build_model(args, mul, id_id, device):
    if args.model == "adapter":
        return BaselineAdapter(
            num_tokens=60,
            d_model=max(args.d_model, 64),
            mlp_layers=args.mlp_layers,
            pool=args.pool,
        ).to(device)

    if args.model == "gru":
        return GRUBaseline(
            num_tokens=60,
            d_model=max(args.d_model, 64),
            num_layers=args.gru_layers,
            dropout=args.gru_dropout,
        ).to(device)

    if args.model == "exact":
        return A5ExactScan(mul_table=mul, id_id=id_id, num_tokens=60).to(device)

    if args.model == "route1":
        return Route1SoftScan(
            mul_table=mul,
            id_id=id_id,
            num_tokens=60,
            temp=args.temp,
            aux_weight=args.aux_weight,
        ).to(device)

    if args.model == "gpt2":
        return GPT2FrozenBaseline(
            num_tokens=60,
            gpt2_name=args.gpt2_name,
            local_files_only=args.local_files_only,
        ).to(device)

    if args.model == "gpt2_state":
        return GPT2FrozenStateFusion(
            mul_table=mul,
            id_id=id_id,
            num_tokens=60,
            gpt2_name=args.gpt2_name,
            inject_layer=args.inject_layer,
            d_state=args.d_state,
            local_files_only=args.local_files_only,
        ).to(device)

    raise ValueError(args.model)


def train_step(model, args, x, y):
    # Optional aux anneal for route1
    if args.model == "route1" and args.anneal_aux:
        # simple piecewise schedule (customize if needed)
        # NOTE: if you want to anneal by global step, implement outside and set model._aux_weight_override.
        pass

    if args.model in {"exact", "route1"}:
        return model(
            x,
            labels=y,
            no_scan=args.no_scan,
            shuffle_M=args.shuffle_M,
            reset_each_step=args.reset_each_step,
        )

    if args.model == "gpt2_state":
        # TRAIN-TIME ablations (keep OFF for Scheme B clean training)
        return model(
            x,
            labels=y,
            shuffle_state=args.shuffle_state,
            reset_state=args.reset_state,
            gate_zero=args.gate_zero,
            state_stride=args.state_stride,
        )

    return model(x, labels=y)


@torch.no_grad()
def run_eval_bundle(model, args, eval_loaders, device, log_path, step, stage_len):
    """
    Scheme B:
      - Always evaluate clean.
      - If args.eval_multi: also evaluate selected eval-only interventions
        (gate_zero / shuffle_state / reset_state) on the same trained model.
    """
    # Define eval configurations
    eval_confs = [("clean", dict(shuffle_state=False, reset_state=False, gate_zero=False))]

    if args.model == "gpt2_state" and args.eval_multi:
        if args.eval_gate_zero:
            eval_confs.append(("gate0", dict(shuffle_state=False, reset_state=False, gate_zero=True)))
        if args.eval_shuffle_state:
            eval_confs.append(("shuffle", dict(shuffle_state=True, reset_state=False, gate_zero=False)))
        if args.eval_reset_state:
            eval_confs.append(("reset", dict(shuffle_state=False, reset_state=True, gate_zero=False)))

    for eval_tag, st_ablate in eval_confs:
        for L, loader in eval_loaders.items():
            acc = eval_final_acc(
                model,
                loader,
                device,
                args.model,
                no_scan=args.no_scan,
                shuffle_M=args.shuffle_M,
                reset_each_step=args.reset_each_step,
                shuffle_state=st_ablate["shuffle_state"],
                reset_state=st_ablate["reset_state"],
                gate_zero=st_ablate["gate_zero"],
                state_stride=args.state_stride,
            )

            print(f"[eval] step {step} | model {args.model} | tag {eval_tag} | len {L} | final_acc {acc:.4f}")

            rec = {
                "step": step,
                "stage_len": stage_len,
                "model": args.model,
                "eval_tag": eval_tag,
                "len": L,
                "final_acc": acc,
                "train_time_ablation": {
                    "no_scan": args.no_scan,
                    "shuffle_M": args.shuffle_M,
                    "reset_each_step": args.reset_each_step,
                    "shuffle_state": args.shuffle_state,
                    "reset_state": args.reset_state,
                    "gate_zero": args.gate_zero,
                },
                "eval_only_ablation": st_ablate,
                "hparams": {
                    "temp": args.temp if args.model == "route1" else None,
                    "aux_weight": args.aux_weight if args.model == "route1" else None,
                    "anneal_aux": args.anneal_aux if args.model == "route1" else None,
                    "gpt2_name": args.gpt2_name if args.model in {"gpt2", "gpt2_state"} else None,
                    "inject_layer": args.inject_layer if args.model == "gpt2_state" else None,
                    "d_state": args.d_state if args.model == "gpt2_state" else None,
                    "local_files_only": args.local_files_only if args.model in {"gpt2", "gpt2_state"} else None,
                },
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    _, mul, id_id = generate_a5()
    device = torch.device(args.device)

    model = build_model(args, mul, id_id, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    eval_lens = [int(x) for x in args.eval_lens.split(",") if x.strip()]
    eval_loaders = {}
    for L in eval_lens:
        ds = RandomSeqFinalDataset(mul, id_id, length=L, num_samples=args.test_samples_per_len, seed=args.seed + 100 + L)
        eval_loaders[L] = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    assert len(schedule) >= 1

    log_path = os.path.join(args.out_dir, "log_final.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    step = 0
    for stage_len in schedule:
        ds_train = RandomSeqFinalDataset(mul, id_id, length=stage_len, num_samples=args.train_samples, seed=args.seed + stage_len)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        print(f"[stage_start] stage_len {stage_len} | steps {args.steps_per_stage}")

        it = iter(train_loader)
        for _ in range(args.steps_per_stage):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            x = batch["input_ids"].to(device)
            y = batch["label_final"].to(device)

            logits, loss = train_step(model, args, x, y)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

            if step % args.log_every == 0:
                print(f"step {step} | stage_len {stage_len} | loss {loss.item():.6f}")

            if step % args.eval_every == 0 and step > 0:
                run_eval_bundle(model, args, eval_loaders, device, log_path, step, stage_len)

            step += 1

        print(f"[stage_end] stage_len {stage_len}")
        run_eval_bundle(model, args, eval_loaders, device, log_path, step, stage_len)

    print(f"Logs written to: {log_path}")


if __name__ == "__main__":
    main()
