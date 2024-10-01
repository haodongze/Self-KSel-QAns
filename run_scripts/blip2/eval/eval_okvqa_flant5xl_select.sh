CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
--cfg-path lavis/projects/blip2/eval/okvqa_flant5xl_select_eval.yaml \
--options run.output_dir="output/blip2_flant5xl/FVQA_select_lora/eval_cycle_select_bs8_lr1e-4_k30_5" \
model.knowledge_num=5 \
model.use_lora=True \
model.load_finetuned=False \
run.batch_size_train=1 \
run.batch_size_eval=1 \
run.init_lr=1e-4 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
model.task='train_cycle_select'