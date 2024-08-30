#new_code == train-squad-agent
run_qa="/home/sqt2/myExperiment/robust_qa/code/run_squad2_multi.py"
trainObjective="New-AGent"
saveModelPath="/reef/sqt2/QA_New_Method/Test_Ablation"
for model in "bert-base-cased" 
do 
    qa_lambda=2
    tagging_lambda=1
    for seed in 1 
    do
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python ${run_qa} \
            --model_name_or_path ${model} \
            --qa_lambda ${qa_lambda} \
            --tagging_lambda ${tagging_lambda} \
            --train_file "/home/sqt2/myExperiment/New_robust_qa/data/training/train-squad-agent.json" \
            --validation_file "/home/sqt2/myExperiment/New_robust_qa/data/dev/SQuAD/dev-v2.0.json" \
            --do_train \
            --do_eval \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 8 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --max_seq_length 384 \
            --max_answer_length 128 \
            --doc_stride 128 \
            --seed ${seed} \
            --save_steps 999999999 \
            --overwrite_output_dir \
            --version_2_with_negative \
            --output_dir  "${saveModelPath}/${trainObjective}/${model}/seed-${seed}"
    done
done