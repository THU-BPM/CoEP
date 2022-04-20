import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers.optimization import AdamW

from src.common.tools import init_logger, logger
from src.common.tools import seed_everything
from src.preprocessing.preprocessor import EnglishPreProcessor
from src.callback.modelcheckpoint import ModelCheckpoint
from src.callback.trainingmonitor import TrainingMonitor
from src.train.metrics import Accuracy, Rouge, F1Score
from src.callback.lr_schedulers import get_linear_schedule_with_warmup

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from configs.basic_config import config
from src.data.utils import collate_fn


def get_clarification_features(args, processor, data, data_split):
    examples = processor.create_examples(
        lines=data,
        data_split=data_split,
        cached_examples_file=config[
                                 'data_dir'] / f"{args.data_name}/cached_examples_clarification_{data_split}",
        clarification=True)
    features = processor.create_features(
        examples=examples,
        max_seq_len=args.max_seq_len,
        cached_features_file=config['data_dir'] / f"{args.data_name}/cached_features_clarification_{data_split}"
                                                  f"{args.max_seq_len}_{args.arch}",
        max_decode_len=args.max_decode_len)

    return features


def get_features(args, processor, data, data_split):
    examples = processor.create_examples(
        lines=data,
        data_split=data_split,
        cached_examples_file=config[
                                 'data_dir'] / f"{args.data_name}/cached_examples_{data_split}")
    features = processor.create_features(
        examples=examples,
        max_seq_len=args.max_seq_len,
        cached_features_file=config['data_dir'] / f"{args.data_name}/cached_features_{data_split}_"
                                                  f"{args.max_seq_len}_{args.arch}",
        max_decode_len=args.max_decode_len)

    return features


def get_fake_features(args, processor, data, data_split):
    examples = processor.create_fake_examples(
        lines=data,
        data_split=data_split,
        cached_examples_file=config[
                                 'data_dir'] / f"{args.data_name}/cached_fake_examples_{data_split}")
    features = processor.create_features(
        examples=examples,
        max_seq_len=args.max_seq_len,
        cached_features_file=config['data_dir'] / f"{args.data_name}/cached_fake_features_"
                                                  f"{data_split}_{args.max_seq_len}_{args.arch}",
        max_decode_len=args.max_decode_len,
        example_type="fake"
    )
    return features


def run_data(args):
    if "conceptnet" in args.data_name:
        from src.data.conceptnet_task_data import TaskData
    elif args.data_name == "v4_atomic":  # "v4_atomic" in args.data_name
        from src.data.atomic_task_data import TaskData
    elif args.data_name == "ART":
        from src.data.art_task_data import TaskData
    else:
        assert args.data_name == "event_story"
        from src.data.story_task_data import TaskData

    train_data = TaskData(raw_data_path=config['data_dir'] / f"{args.data_name}/{args.data_name}_train.csv",
                          preprocessor=EnglishPreProcessor(),
                          is_train=True)
    train_targets, train_sentences, train_chars = train_data.read_data()
    train_data.save_data(X=train_sentences,
                         y=train_targets,
                         c=train_chars,
                         shuffle=True,
                         data_name=args.data_name,
                         data_dir=config['data_dir'] / f"{args.data_name}",
                         data_split='train')
    valid_data = TaskData(raw_data_path=config['data_dir'] / f"{args.data_name}/{args.data_name}_valid.csv",
                          preprocessor=EnglishPreProcessor(),
                          is_train=True)
    valid_targets, valid_sentences, valid_chars = valid_data.read_data()
    valid_data.save_data(X=valid_sentences,
                         y=valid_targets,
                         c=valid_chars,
                         shuffle=True,
                         data_name=args.data_name,
                         data_dir=config['data_dir'] / f"{args.data_name}",
                         data_split='valid')


def load_train_valid_data(args, tokenizer):
    if args.data_name == "event_story":
        from src.data.bart_processor_story import BartProcessor
    elif args.data_name == "ART":
        from src.data.bart_processor_art import BartProcessor
    else:
        from src.data.bart_processor import BartProcessor
    processor = BartProcessor(tokenizer)
    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}/{args.data_name}.train.pkl")
    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}/{args.data_name}.valid.pkl")

    if "clarification" in args.model_name:
        train_features = get_clarification_features(args, processor, train_data, "train")
        valid_features = get_clarification_features(args, processor, valid_data, "valid")
    else:
        train_features = get_features(args, processor, train_data, "train")
        valid_features = get_features(args, processor, valid_data, "valid")
        if args.add_cls:
            fake_train_features = get_fake_features(args, processor, train_data, "train")
            fake_valid_features = get_fake_features(args, processor, valid_data, "valid")

            train_features += fake_train_features
            valid_features += fake_valid_features

    train_features = train_features
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)

    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    valid_features = valid_features
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)

    logger.info("***** Running training: Loading data *****")
    logger.info("  Train examples = %d", len(train_features))
    logger.info("  Valid examples = %d", len(valid_features))
    return train_dataloader, valid_dataloader


def run_train(args):
    tokenizer = BartTokenizer.from_pretrained(str(config[f'{args.arch}_model_dir']))
    # ------- data
    train_dataloader, valid_dataloader = load_train_valid_data(args, tokenizer)
    # ------- model
    logger.info("initializing model")
    bart_config = BartConfig.from_pretrained(str(config[f'{args.arch}_config_path']))

    if args.resume_path:
        states = torch.load(f"{args.resume_path}/epoch_{args.resume_epoch}_bart-base_model.bin")
        logger.info(f"Load model from {args.resume_path} with epoch {states['epoch']}")
    else:
        states = None

    if args.add_cls:
        from src.model.bart_for_generation import BartWithClassification
        model = BartWithClassification(path=str(config[f'{args.arch}_model_dir']), num_labels=2)
    else:
        model = BartForConditionalGeneration.from_pretrained(str(config[f'{args.arch}_model_dir']),
                                                             config=bart_config)
    if states:
        model.load_state_dict(states["model"])

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ---- callbacks
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

    loss_ignore_index = tokenizer.pad_token_id if args.skip_pad_loss else -100

    if args.add_cls:
        from src.train.trainer_bart import TrainerCls
        epoch_metrics = [Rouge(), F1Score(task_type='multiclass', average='macro')]
        trainer = TrainerCls(args=args,
                             model=model,
                             tokenizer=tokenizer, logger=logger, optimizer=optimizer, scheduler=scheduler,
                             early_stopping=None, training_monitor=train_monitor, model_checkpoint=model_checkpoint,
                             epoch_metrics=epoch_metrics,
                             batch_metrics=[Accuracy(topK=1, ignore_index=loss_ignore_index)],
                             loss_ignore_index=loss_ignore_index)
    else:
        from src.train.trainer_bart import Trainer
        trainer = Trainer(args=args,
                          model=model,
                          tokenizer=tokenizer, logger=logger, optimizer=optimizer, scheduler=scheduler,
                          early_stopping=None, training_monitor=train_monitor, model_checkpoint=model_checkpoint,
                          epoch_metrics=[Rouge()],
                          batch_metrics=[Accuracy(topK=1, ignore_index=loss_ignore_index)],
                          loss_ignore_index=loss_ignore_index)

    # see the resource of trainer to find the progress of metric calculation!!!
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bart-base', type=str)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument('--resume_epoch', default=15, type=int)
    parser.add_argument('--data_name', default='v4_atomic', type=str)
    parser.add_argument("--model_name", default='', type=str)

    parser.add_argument("--add_cls", action='store_true')
    parser.add_argument("--skip_pad_loss", action="store_true")

    parser.add_argument("--do_data", action='store_true')  # for v4_atomic, change the relations for need
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--epochs", default=15, type=int)

    parser.add_argument("--valid_size", default=0.15, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=0, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='1', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument("--max_seq_len", default=150, type=int)
    parser.add_argument("--max_decode_len", default=20, type=int)

    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
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
    torch.save(args, f'{config["checkpoint_dir"]}/training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        run_data(args)

    if args.do_train:
        run_train(args)


if __name__ == '__main__':
    print(f"begin run.......")
    main()
