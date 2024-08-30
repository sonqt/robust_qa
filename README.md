# Robust Extractive Question Answering
This is an official implementation for our paper, [Towards Robust Extractive Question Answering Models: Rethinking the Training Methodology
](https://openreview.net/forum?id=LlUCoqtgpd), 2024. 

**Acknowledgement**: Our implementation and this README primarily builds upon the [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) script provided by HuggingFace. 

```bibtex
bibtex will available soon
```
### Fine-tuning BERT on SQuAD1.1/SQuAD 2.0

The [`robust_run_qa.py`](https://github.com/sonqt/robust_qa/blob/main/code/robust_qa_models.py) script allows to fine-tune model (as long as its architecture has a `Robust[Model]ForQuestionAnswering` version in the `robust_qa_models.py` script) on a question-answering dataset (such as SQuAD, or any other QA dataset available in the `datasets` library, or your own csv/jsonlines files) as long as they are structured the same way as SQuAD. You might need to tweak the data processing inside the script if your data is structured differently.

Our implementation currently supports BERT, RoBERTa, and SpanBERT. If you require support for additional models, please open an issue.

```bash
run_qa="code/run_squad2_multi.py"
save_path="PATH-TO-SAVE-MODEL-AND-RESULTS"
train_path="NAME/PATH-OF-TRAIN-DATA"
validation_path="NAME/PATH-OF-VALIDATION-DATA"
model="bert-base-cased"  
qa_lambda=2
tagging_lambda=1
seed=1 
python ${run_qa} \
    --model_name_or_path ${model} \
    --qa_lambda ${qa_lambda} \
    --tagging_lambda ${tagging_lambda} \
    --train_file ${train_path} \
    --validation_file ${validation_path} \
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
    --overwrite_output_dir \
    --output_dir  "${save_path}"
```

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag `--version_2_with_negative`.

```bash
python ${run_qa} \
    --model_name_or_path ${model} \
    --qa_lambda ${qa_lambda} \
    --tagging_lambda ${tagging_lambda} \
    --train_file ${train_path} \
    --validation_file ${validation_path} \
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
    --overwrite_output_dir \
    --version_2_with_negative \
    --output_dir  "${save_path}"
```