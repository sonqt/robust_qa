# run_qa="/Volumes/Share/tran_s2/SQuAD2_new/code/run_squad2_multi.py"
# data_path="/Volumes/Share/tran_s2/SQuAD2_new/data/shortcut-study/multi-ans"
# portion="1"
# trainObjective="new-multians"
# saveModelPath="/Volumes/Share/tran_s2/SQuAD2_new/shortcut-study"
# model="bert-base-cased"
# python ${run_qa} \
#     --model_name_or_path ${model} \
#     --train_file "${data_path}/${portion}.json" \
#     --validation_file "/Volumes/Share/tran_s2/Public_Datasets/SQuAD/squad2/dev-v2.0-new-format.json" \
#     --do_train \
#     --do_eval \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 8 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 2 \
#     --max_seq_length 384 \
#     --max_answer_length 128 \
#     --doc_stride 128 \
#     --save_steps 999999 \
#     --overwrite_output_dir \
#     --version_2_with_negative \
#     --output_dir  "${saveModelPath}/${trainObjective}/${model}/${portion}"
MODEL_PATH="/Volumes/Share/tran_s2/SQuAD2_new/shortcut-study"
trainObjective="new-multians"
EVAL_DATA="/Volumes/Share/tran_s2/Public_Datasets"
SAVE_PATH="/Volumes/Share/tran_s2/SQuAD2_new/shortcut-study/Prediction"
model="bert-base-cased"
for portion in 1 2 3 4 5
do 
    for data in SQuAD/squad-attack/dev_adversarial_attack SQuAD/squad-attack/dev_negation_attack
    do
        python /Volumes/Share/tran_s2/SQuAD2_new/code/run_squad2_multi.py \
            --model_name_or_path "${MODEL_PATH}/${trainObjective}/${model}/${portion}" \
            --validation_file "${EVAL_DATA}/${data}.json" \
            --do_eval \
            --per_device_eval_batch_size 8 \
            --max_seq_length 512 \
            --max_answer_length 128 \
            --doc_stride 128 \
            --n_best_size 5 \
            --overwrite_output_dir \
            --version_2_with_negative \
            --output_dir "${SAVE_PATH}/${trainObjective}/${model}/${portion}/${data}"
    done
done