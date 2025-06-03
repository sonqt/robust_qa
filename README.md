<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Towards Robust Extractive Question Answering Models: Rethinking the Training Methodology </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://sonqt.github.io" target="_blank" style="text-decoration: none;">Son Quoc Tran<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="http://personal.denison.edu/~kretchmar/" target="_blank" style="text-decoration: none;">Matt Kretchmar<sup>2</sup></a>&nbsp;,&nbsp;
    <br/> 
    <sup>1</sup>Cornell University&nbsp;&nbsp;&nbsp;<sup>2</sup>Denison University<br> 
</p>
    
This is an official implementation for our paper, [Towards Robust Extractive Question Answering Models: Rethinking the Training Methodology
](https://openreview.net/forum?id=LlUCoqtgpd), 2024. 

**Acknowledgement**: Our implementation and this README primarily build upon the [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) script provided by HuggingFace. 

```bibtex
@inproceedings{tran-kretchmar-2024-towards,
    title = "Towards Robust Extractive Question Answering Models: Rethinking the Training Methodology",
    author = "Tran, Son  and
      Kretchmar, Matt",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.121",
    pages = "2222--2236",
    abstract = "This paper proposes a novel training method to improve the robustness of Extractive Question Answering (EQA) models. Previous research has shown that existing models, when trained on EQA datasets that include unanswerable questions, demonstrate a significant lack of robustness against distribution shifts and adversarial attacks. Despite this, the inclusion of unanswerable questions in EQA training datasets is essential for ensuring real-world reliability. Our proposed training method includes a novel loss function for the EQA problem and challenges an implicit assumption present in numerous EQA datasets. Models trained with our method maintain in-domain performance while achieving a notable improvement on out-of-domain datasets. This results in an overall F1 score improvement of 5.7 across all testing sets. Furthermore, our models exhibit significantly enhanced robustness against two types of adversarial attacks, with a performance decrease of only about one-third compared to the default models.",
}

```
### Fine-tuning BERT on SQuAD1.1/SQuAD 2.0

The [`robust_run_qa.py`](https://github.com/sonqt/robust_qa/blob/main/code/robust_qa_models.py) script allows to fine-tune model (as long as its architecture has a `Robust[Model]ForQuestionAnswering` version in the `robust_qa_models.py` script) on a question-answering dataset (such as SQuAD, or any other QA dataset available in the `datasets` library, or your own csv/jsonlines files) as long as they are structured the same way as SQuAD. You might need to tweak the data processing inside the script if your data is structured differently.

Our implementation currently supports BERT, RoBERTa, and SpanBERT. If you require support for additional models, please open an issue.

```bash
run_qa="code/robust_run_qa.py"
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
### Fine-tuning BERT on SQuAD 2.0 Multians
Our training data is available at [datasets/sonquoctran/Multians](https://huggingface.co/datasets/sonquoctran/Multians).
