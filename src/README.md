To run future event generation experiment, follow there instructions:

## First Step
Prepare pre-trained models (which will be published after reviewing process).

## Then install dependencies

```
pytorch == 1.7.1
transformers = 4.2.1
```
## Fine-tuning IM and GM with ATOMIC and ConceptNet
Run the script to train IM with the additional CLS HEAD on ATOMIC and do evaluations

```python train_bart_model.py --data_name v4_atomic --add_cls```

Run the script to train GM on sequential ConceptNet and do evaluations

```python train_bart_model.py --data_name conceptnet_seq```

Test the fine-tuned model by running the following script

```python test_bart_model.py --data_name v4_atomic/conceptnet_seq --resume_path MODEL_PATH --resume_epoch MODEL_ITER```


## Train ECoGM with fine-tuned models
Run the following script to train ECoGM on fine-tuned IM and GM

```python train_ECoGM.py --load_kg_model_with_cls --pretrain_GM --data_name event_story```

Run the script to test the performance of future event generation on CommonEvent

```python test_ECoGM.py --data_name event_story --train_data event_story --resume_path MODEL_PATH --resume_epoch MODEL_ITER```

## Playing Around in Interactive Mode

Run the script to interavtive input context and event in the terminal
and generate future events

```python interactive_generate.py```
