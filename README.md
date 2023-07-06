# BERT-based NER using HuggingFace Transformers

The model is deployed on HuggingFace Hub and can be accessed [here](https://huggingface.co/MUmairAB/bert-ner).

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on [Cnoll2003](https://huggingface.co/datasets/conll2003) dataset.
It achieves the following results on the evaluation set:
- Train Loss: 0.0003
- Validation Loss: 0.0880
- Epoch: 19

## How to use this model

```
#Install the transformers library
!pip install transformers

#Import the pipeline
from transformers import pipeline

#Import the model from HuggingFace
checkpoint = "MUmairAB/bert-ner"
model = pipeline(task="token-classification",
                 model=checkpoint)

#Use the model
raw_text = "My name is umair and i work at Swits AI in Antarctica."
model(raw_text)

```

## Model description

Model: "tf_bert_for_token_classification"
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 bert (TFBertMainLayer)      multiple                  107719680 
                                                                 
 dropout_37 (Dropout)        multiple                  0         
                                                                 
 classifier (Dense)          multiple                  6921      
                                                                 
=================================================================
Total params: 107,726,601
Trainable params: 107,726,601
Non-trainable params: 0
_________________________________________________________________
```
## Framework versions

- Transformers 4.30.2
- TensorFlow 2.12.0
- Datasets 2.13.1
- Tokenizers 0.13.3

