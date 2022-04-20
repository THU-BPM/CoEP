import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers.optimization import AdamW

from src.common.tools import init_logger, logger
from src.common.tools import seed_everything
from src.preprocessing.preprocessor import EnglishPreProcessor
from src.callback.modelcheckpoint import ModelCheckpoint
from src.callback.trainingmonitor import TrainingMonitor
from src.callback.lr_schedulers import get_linear_schedule_with_warmup

from configs.basic_config import config
from src.train.metrics import Accuracy, Rouge, F1Score

from src.data.utils import collate_fn as collate_fn
from src.model.CoEP_Model import CoEP as Model
from src.model.bart_for_generation import BartWithClassification
from configs.finetuning_config import config as ft_config


def get_features(processor, data, args, data_split, example_type=None):
    if example_type is None:
        cached_examples_file = config['data_dir'] / f"{args.data_name}/cached_examples_{data_split}"
        examples = processor.create_examples(
            lines=data,
            data_split=data_split,
            cached_examples_file=cached_examples_file
        )
    else:
        cached_examples_file = config['data_dir'] / f"{args.data_name}/cached_examples_{data_split}_{example_type}"
        examples = processor.create_fake_examples(
            lines=data,
            data_split=data_split,
            cached_examples_file=cached_examples_file
        )

    if example_type is None:
        cached_features_file = config['data_dir'] / f"{args.data_name}/cached_{data_split}_features_{args.data_name}_" \
                                                    f"{args.max_seq_len}_{args.arch}"
    else:
        cached_features_file = config['data_dir'] / f"{args.data_name}/cached_{data_split}_{example_type}_features_" \
                                                    f"{args.data_name}_{args.max_seq_len}_{args.arch}"
    features = processor.create_features(
        examples=examples,
        max_seq_len=args.max_seq_len,
        cached_features_file=cached_features_file,
        max_decode_len=args.max_decode_len,
        his_len=args.history_length,
        example_type=example_type
    )
    return features


def run_data(args, data_split):
    if args.data_name == "event_story":
        from src.data.story_task_data import TaskData
    else:
        from src.data.art_task_data import TaskData

    data = TaskData(raw_data_path=config['data_dir'] / f"{args.data_name}/{args.data_name}_{data_split}.csv",
                    preprocessor=EnglishPreProcessor(),
                    is_train=True)
    targets, sentences, chars = data.read_data()
    data.save_data(X=sentences,
                   y=targets,
                   c=chars,
                   shuffle=True,
                   data_name=args.data_name,
                   data_dir=config['data_dir'] / f"{args.data_name}",
                   data_split=data_split)


def load_data(args, tokenizer, data_split):
    if args.data_name == "event_story":
        from src.data.bart_processor_story import BartProcessor
    else:
        from src.data.bart_processor_art import BartProcessor

    processor = BartProcessor(tokenizer=tokenizer)
    data = processor.get_train(config['data_dir'] / f"{args.data_name}/{args.data_name}.{data_split}.pkl")
    features = get_features(processor, data, args, data_split=data_split)
    if args.add_cls:
        fake_features = get_features(processor, data, args, data_split=data_split, example_type="fake")
        features.extend(fake_features)

    features = features
    dataset = processor.create_dataset(features, is_sorted=args.sorted)

    if data_split == "train":
        sampler = SequentialSampler(dataset) if args.sorted else RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    logger.info(f"***** Running training: Loading {data_split} data *****")
    logger.info("  Num features = %d", len(features))

    return dataloader


def load_atomic_KG_model(arch, n_gpu):
    resume_path = ft_config["atomic_kg_wcls_model"]
    IM = BartWithClassification(path=config[f'{arch}_model_dir'], num_labels=2)
    IM.load_state_dict(torch.load(resume_path, map_location=f"cuda:{n_gpu}")["model"])
    return IM


def load_sequential_KG_model(arch, n_gpu, pretrain_GM=None):
    model = BartForConditionalGeneration.from_pretrained(str(config[f'{arch}_model_dir']))
    if pretrain_GM:
        resume_path = ft_config["conceptnet_seq_model"]
        model.load_state_dict(torch.load(resume_path, map_location=f"cuda:{n_gpu}")["model"])
    return model


def run_train(args):
    tok = BartTokenizer.from_pretrained(str(config[f'{args.arch}_model_dir']))
    # --------- data --------
    train_dataloader = load_data(args, tokenizer=tok, data_split="train")
    valid_dataloader = load_data(args, tokenizer=tok, data_split="valid")
    # ------- model ---------
    logger.info("initializing model")

    num_labels = 2 if args.add_cls else None
    kg_model = load_atomic_KG_model(args.arch, args.n_gpu)

    IM = kg_model.model
    GM = load_sequential_KG_model(args.arch, args.n_gpu, pretrain_GM=args.pretrain_GM)
    bart_config = BartConfig.from_pretrained(config[f'{args.arch}_config_path'])

    model = Model(bart_config=bart_config, pretrained_im=IM, pretrained_gm=GM, num_labels=None, fixed_kg=False)

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # **************************** callbacks ***************************
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'] / args.model_name, arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'], mode=args.mode,
                                       monitor=args.monitor, arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training: Training args *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    loss_ignore_index = tok.pad_token_id if args.skip_pad_loss else -100

    # see the resource of trainer to find the progress of metric calculation!!!
    logger.info("Use gumbel softmax to generate the explanations.")
    if args.add_cls:
        from src.train.trainer_CoEP import TrainerCls
        trainer = TrainerCls(args=args,
                             model=model,
                             kg_model=kg_model,
                             tok=tok, logger=logger, optimizer=optimizer, scheduler=scheduler,
                             early_stopping=None, training_monitor=train_monitor,
                             model_checkpoint=model_checkpoint,
                             epoch_metrics=[Rouge(), F1Score(task_type='multiclass', average='macro')],
                             batch_metrics=[Accuracy(topK=1, ignore_index=loss_ignore_index)],
                             loss_ignore_index=loss_ignore_index)
    else:
        from src.train.trainer_CoEP import Trainer
        trainer = Trainer(args=args,
                          model=model,
                          kg_model=kg_model,
                          tok=tok, logger=logger, optimizer=optimizer, scheduler=scheduler,
                          early_stopping=None, training_monitor=train_monitor,
                          model_checkpoint=model_checkpoint,
                          epoch_metrics=[Rouge()],
                          batch_metrics=[Accuracy(topK=1, ignore_index=loss_ignore_index)],
                          loss_ignore_index=loss_ignore_index)
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bart-base', type=str)
    parser.add_argument('--data_name', default='event_story', type=str) # ART
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--resume_epoch', default=4, type=int)
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--history_length", type=int, default=100)
    parser.add_argument("--length_penalty", type=float, default=1.0)

    parser.add_argument("--load_kg_model_with_cls", action="store_true")
    parser.add_argument("--pretrain_GM", action="store_true")

    parser.add_argument("--training_explanation", action="store_true")
    parser.add_argument("--explanation_coeff", type=float, default=10.0)
    parser.add_argument("--explanation_len", type=int, default=10)  # max 10 for faster inference
    parser.add_argument("--use_kld_loss", action="store_true")
    parser.add_argument("--use_cls_loss", action="store_true")

    parser.add_argument("--add_cls", action="store_true")
    parser.add_argument("--skip_pad_loss", action="store_true")

    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--max_decode_len", default=20, type=int)

    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--valid_size", default=0.15, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=0, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    init_logger(
        log_file=f'{config["log_dir"]}/{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}-{args.model_name}.log')
    import os

    config["checkpoint_dir"] = f'{config["checkpoint_dir"]}/{args.arch}_{args.data_name}_{args.model_name}'
    if not os.path.isdir(config["checkpoint_dir"]):
        os.mkdir(config["checkpoint_dir"])
    # Good practice: save your training arguments together with the trained model
    torch.save(args, f'{config["checkpoint_dir"]}/training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_data:
        run_data(args, data_split="train")
        run_data(args, data_split="valid")

    if args.do_train:
        run_train(args)


if __name__ == '__main__':
    main()
