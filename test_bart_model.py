import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import SequentialSampler

from src.data.utils import collate_fn as collate_fn
from src.common.tools import init_logger, logger
from configs.basic_config import config
from src.preprocessing.preprocessor import EnglishPreProcessor
from src.test.sampler import BeamSampler, TopKSampler, GreedySampler
from src.test.generator_bart import Generator


def set_sampler(cfg, sampling_algorithm):
    if "beam" in sampling_algorithm:
        cfg.set_numbeams(int(sampling_algorithm.split("-")[1]))
        sampler = BeamSampler(cfg)
    elif "topk" in sampling_algorithm:
        # print("Still bugs in the topk sampler. Use beam or greedy instead")
        # raise NotImplementedError
        cfg.set_topK(int(sampling_algorithm.split("-")[1]))
        sampler = TopKSampler(cfg)
    else:
        sampler = GreedySampler(cfg)
    return sampler


def run_test(args):
    if "conceptnet" in args.data_name:
        from src.data.conceptnet_task_data import TaskData
        from src.data.bart_processor import BartProcessor
    elif args.data_name == "v4_atomic":  # "v4_atomic" in args.data_name
        from src.data.atomic_task_data import TaskData
        from src.data.bart_processor import BartProcessor
    else:
        from src.data.story_task_data import TaskData
        from src.data.bart_processor_story import BartProcessor

    data = TaskData(raw_data_path=f'dataset/{args.data_name}/{args.data_name}_test_refs.csv',
                    preprocessor=EnglishPreProcessor(),
                    is_train=False)

    targets, sentences, relations = data.read_data()
    lines = list(zip(sentences, targets, relations))

    print(f"num of lines: {len(lines)}")

    tok = BartTokenizer.from_pretrained(str(config[f'{args.arch}_model_dir']))
    processor = BartProcessor(tok)

    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(
        lines=test_data,
        data_split='test',
        cached_examples_file=config[
                                 'data_dir'] / f"{args.data_name}/cached_test_examples")
    test_features = processor.create_features(
        examples=test_examples,
        max_seq_len=args.max_seq_len,
        cached_features_file=config['data_dir'] / f"{args.data_name}/cached_test_features_{args.max_seq_len}_{args.arch}",
        max_decode_len=args.max_decode_len)
    test_features = test_features

    print(f"num of examples: {len(test_features)}")
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

    states = torch.load(f'{args.resume_path}/epoch_{args.resume_epoch}_{args.arch}_model.bin',
                        map_location=f"cuda:{args.n_gpu}")
    logger.info(f"Load model from {args.resume_path} with epoch {states['epoch']}")

    if args.add_cls:
        from src.model.bart_for_generation import BartWithClassification
        model = BartWithClassification(path=str(config[f'{args.arch}_model_dir']), num_labels=2)
    else:
        model = BartForConditionalGeneration.from_pretrained(str(config[f'{args.arch}_model_dir']))

    model.load_state_dict(states["model"])

    logger.info('model predicting....')

    lp = args.length_penalty if args.length_penalty is not None else 1.0
    # num_return_sequences = int(args.sampling_algorithm.split("-")[1])
    num_return_sequences = 1

    if "cls" in args.log_info:
        from src.test.sampler_config import CFG
        cfg = CFG(args.max_decode_len, tok)
        assert args.sampling_algorithm is not None
        sampler = set_sampler(cfg, args.sampling_algorithm)
        generator = Generator(model=model, tok=tok, logger=logger, n_gpu=args.n_gpu)
        label_sents, generated_sents, encoder_states, cls_ids = generator.generate_with_cls(
            data=test_dataloader,
            sampler=sampler,
            max_length=args.max_decode_len,
            length_penalty=lp,
            save_prefix=f"{args.log_info}_{args.sampling_algorithm}_lp_{lp}.pkl",
            num_return_sequences=num_return_sequences,
            add_cls=True)
    else:
        generator = Generator(model=model, tok=tok, logger=logger, n_gpu=args.n_gpu)
        label_sents, generated_sents, encoder_states_list = generator.generate(
            data=test_dataloader,
            max_length=args.max_decode_len,
            length_penalty=lp,
            num_beams=4,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            save_file=f"{args.log_info}_lp_{lp}.csv")


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bart-base', type=str)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--resume_epoch", type=int, default=4)
    parser.add_argument('--data_name', default='event_story', type=str)
    parser.add_argument("--log_info", default="", type=str)
    parser.add_argument("--add_cls", action="store_true")

    parser.add_argument("--length_penalty", default=2.0, type=float)
    parser.add_argument("--sampling_algorithm", default='beam-4', type=str)

    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--max_decode_len", default=20, type=int)
    parser.add_argument("--n_gpu", type=str, default='5', help='"0,1,.." or "0" or "" ')

    args = parser.parse_args()

    init_logger(
        log_file=config['log_dir'] / f'{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}-{args.log_info}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'testing_args.bin')
    logger.info("Testing parameters %s", args)

    run_test(args)


if __name__ == '__main__':
    main()
