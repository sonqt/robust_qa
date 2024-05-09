# run_qa="/Volumes/Share/tran_s2/SQuAD2_new/code/run_squad2_multi.py"
# trainObjective="SQuAD-AGent-multians"
# saveModelPath="/Volumes/Share/tran_s2/SQuAD2_new/Model"
# model="roberta-base"
# python ${run_qa} \
#     --model_name_or_path ${model} \
#     --train_file "/Volumes/Share/tran_s2/SQuAD2_new/data/train-squad-agent-multians.json" \
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
#     --output_dir  "${saveModelPath}/${trainObjective}/${model}"

MODEL_PATH="/Volumes/Share/tran_s2/SQuAD2_new/Model"
trainObjective="SQuAD-AGent-multians"
EVAL_DATA="/Volumes/Share/tran_s2/Public_Datasets"
SAVE_PATH="/Volumes/Share/tran_s2/SQuAD2_new/Prediction"

for model in roberta-base
do 
    for data in AGent/SQuAD_AGent_dev AGent/HotpotQA_AGent_dev AGent/HotpotQA_answerable_dev AGent/NQ_answerable_dev AGent/NQ_unanswerable_dev ACE/ACE-whQA-competitive ACE/ACE-whQA-noncompetitive ACE/ACE-whQA-has-answer SQuAD/squad-attack/dev_adversarial_attack SQuAD/squad-attack/dev_negation_attack
    do
        python /Volumes/Share/tran_s2/SQuAD2_new/code/run_squad2_multi.py \
            --model_name_or_path "${MODEL_PATH}/${trainObjective}/${model}" \
            --validation_file "${EVAL_DATA}/${data}.json" \
            --do_eval \
            --per_device_eval_batch_size 8 \
            --max_seq_length 512 \
            --max_answer_length 128 \
            --doc_stride 128 \
            --n_best_size 5 \
            --overwrite_output_dir \
            --version_2_with_negative \
            --output_dir "${SAVE_PATH}/${trainObjective}/${model}/${data}"
    done
done