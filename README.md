---
base_model: openai/whisper-small
language:
- ar
license: apache-2.0
metrics:
- wer
tags:
- generated_from_trainer
model-index:
- name: Tunisian Checkpoint12
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: custom_tunisian_dataset
      type: dataset
      args: 'config: ar, split: test'
    metrics:
    - type: wer
      value: 54.52300785634119
      name: Wer
    - type: cer
      value: 25.538666370797735
      name: Cer
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->


# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



### Finetuning Whisper on Tunisian custom dataset

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the tunisian_custom dataset =more than 4h(/doumawl4+/doumaw02+Data3+dataset1+dataset2).
It achieves the following results on the evaluation set:
- Train Loss: 0.0109
- Evaluation Loss: 1.1608097553253174
- Wer:  54.52300785634119
- Cer:  25.538666370797735
-max_audio_length=15
for the preprocessing i used padding+VAD filter
- **Developed by:** [Ameni Khabthani]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [ASR system]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [whisper small]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]


#### Training Hyperparameters

per_device_train_batch_size=4       
gradient_accumulation_steps=8        
learning_rate=  5e-5    

warmup_steps=100                      
max_steps=4000                       
gradient_checkpointing=True           
fp16=True      
save_steps=500                        
eval_steps=500                        
per_device_eval_batch_size=8          
predict_with_generate=True             
generation_max_length=249     
logging_steps=50                     
weight_decay=0.001
dropout=0.1
optim="adamw_bnb_8bit"                 
seed=42
save_total_limit=5
save_steps=500,                         
eval_steps=500, 

