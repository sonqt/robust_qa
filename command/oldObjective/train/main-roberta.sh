trainObjective="SQuADv2-old"
saveModelPath="/Volumes/Share/tran_s2/SQuAD2_new/Models"
model="roberta-base"
for seed in 1 2 3 4 5
do
    python /Volumes/Share/tran_s2/QuestionAnswering/code/run_qa.py \
        --model_name_or_path ${model} \
        --train_file "/Volumes/Share/tran_s2/Public_Datasets/SQuAD/squad2/train-v2.0-new-format.json" \
        --validation_file "/Volumes/Share/tran_s2/Public_Datasets/SQuAD/squad2/dev-v2.0-new-format.json" \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --max_seq_length 384 \
        --max_answer_length 128 \
        --doc_stride 128 \
        --save_steps 999999999 \
        --eval_steps 999999999 \
        --seed ${seed} \
        --overwrite_output_dir \
        --version_2_with_negative \
        --output_dir  "${saveModelPath}/${trainObjective}/${model}/seed-${seed}"
done