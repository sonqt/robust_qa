MODEL_PATH="/Volumes/Share/tran_s2/SQuAD2_new/Models"
trainObjective="SQuADv2-multians"
EVAL_DATA="/Volumes/Share/tran_s2/Public_Datasets"
SAVE_PATH="/Volumes/Share/tran_s2/SQuAD2_new/Progress"
model="bert-base-cased"
for seed in 1 2 3 4 5
do 
for cp in 2478 4956 7434 9912 12390 14868 17346 19824 22302
do
    for data in AGent/SQuAD_AGent_dev SQuAD/squad2/dev-v2.0-new-format SQuAD/squad-attack/dev_adversarial_attack SQuAD/squad-attack/dev_negation_attack
    do
        python /Volumes/Share/tran_s2/SQuAD2_new/code/run_squad2_multi.py \
            --model_name_or_path "${MODEL_PATH}/${trainObjective}/${model}/seed-${seed}/checkpoint-${cp}" \
            --validation_file "${EVAL_DATA}/${data}.json" \
            --do_eval \
            --per_device_eval_batch_size 8 \
            --max_seq_length 512 \
            --max_answer_length 128 \
            --doc_stride 128 \
            --n_best_size 5 \
            --overwrite_output_dir \
            --version_2_with_negative \
            --output_dir "${SAVE_PATH}/${trainObjective}/${model}/seed-${seed}/checkpoint-${cp}/${data}"
    done
done
done