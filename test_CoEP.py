import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration

from src.common.tools import init_logger, logger
from configs.basic_config import config
from src.preprocessing.preprocessor import EnglishPreProcessor

from src.test.sampler import GreedySampler, TopKSampler, BeamSampler
from src.test.sampler_config import CFG
from configs.finetuning_config import config as ft_config
from src.data.bart_processor_story import BartProcessor
from src.data.utils import collate_fn as collate_fn
from src.data.story_task_data import TaskData
from src.model.bart_for_generation import BartWithClassification
from src.model.CoEP_Model import ECoGM


def load_atomic_KG_wcls_model(arch, n_gpu):
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


def set_sampler(cfg, sampling_algorithm):
    if "beam" in sampling_algorithm:
        logger.info(f"Initialize Beam sampling!")
        cfg.set_numbeams(int(sampling_algorithm.split("-")[1]))
        sampler = BeamSampler(cfg)
    elif "topk" in sampling_algorithm:
        # print("Still bugs in the topk sampler. Use beam or greedy instead")
        # raise NotImplementedError
        logger.info(f"Initialize TopK sampling!")
        cfg.set_topK(int(sampling_algorithm.split("-")[1]))
        sampler = TopKSampler(cfg)
    else:
        sampler = GreedySampler(cfg)

    return sampler


def load_test_data(args, tokenizer):
    data = TaskData(raw_data_path=config["data_dir"] / f'{args.data_name}/{args.data_name}_test_refs.csv',
                    preprocessor=EnglishPreProcessor(),
                    is_train=False)
    targets, sentences, chars = data.read_data()
    lines = list(zip(sentences, targets, chars))

    processor = BartProcessor(tokenizer=tokenizer)
    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(
        lines=test_data,
        data_split='test',
        cached_examples_file=config['data_dir'] / f"{args.data_name}/cached_test_examples")
    test_features = processor.create_features(
        examples=test_examples,
        max_seq_len=args.max_seq_len,
        cached_features_file=config[
                                 'data_dir'] / f"{args.data_name}/cached_test_features_{args.max_seq_len}_{args.arch}",
        max_decode_len=args.max_decode_len,
        his_len=args.history_length,
    )

    test_features = test_features
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

    logger.info('Loading Test Data....')
    logger.info("  Num examples = %d", len(test_features))

    return test_dataloader


def run_test(args):
    states = torch.load(f'{args.resume_path}/epoch_{args.resume_epoch}_{args.arch}_model.bin',
                        map_location=f"cuda:{args.n_gpu}")

    tok = BartTokenizer.from_pretrained(str(config[f'{args.arch}_model_dir']))
    test_dataloader = load_test_data(args, tok)

    num_lables = 2 if args.add_cls else None

    model_state_dict = states['model'] if 'model' in states else states
    kg_bart_model = load_atomic_KG_wcls_model(args.arch, args.n_gpu)
    model = ECoGM(kg_bart_model=kg_bart_model.model,
                  bart_model=load_sequential_KG_model(args.arch, args.n_gpu),
                  num_labels=num_lables)
    model.load_state_dict(model_state_dict)

    # ----------- predicting ----------
    logger.info('model predicting....')
    cfg = CFG(args.max_decode_len, tok)
    sampler = set_sampler(cfg, args.sampling_algorithm)

    from src.test.generator_CoEP import Generator
    generator = Generator(model=model, tok=tok, logger=logger, n_gpu=args.n_gpu)

    lp = args.length_penalty
    # num_return_sequences = int(args.sampling_algorithm.split("-")[1])
    num_return_sequences = 1
    relation_index = args.relation_index

    if args.generate_explanation:
        kg_model = load_atomic_KG_wcls_model(args.arch, args.n_gpu)
        logger.info(f"Generate explanations")
        label_sents, generated_sents, _, _ = generator.generate_explanation(
            data=test_dataloader,
            KG_model=kg_model,
            sampler=sampler,
            max_length=args.max_decode_len,
            repetition_penalty=None,
            length_penalty=lp,
            relation_index=relation_index,
            no_repeat_ngram_size=None,
            save_prefix=f"test_case_{args.data_name}_{args.log_info}_{args.sampling_algorithm}_lp{lp}.csv",
            KG_only=True,
            num_return_sequences=num_return_sequences)
    else:
        label_sents, generated_sents, _, _ = generator.generate_example(
            data=test_dataloader,
            sampler=sampler,
            max_length=args.max_decode_len,
            repetition_penalty=None,
            length_penalty=lp,
            relation_index=relation_index,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=None,
            save_prefix=f"test_case_{args.data_name}_{args.log_info}_lp{lp}.csv",
            add_cls=args.add_cls)


def main():
    parser = ArgumentParser()
    parser.add_argument('--arch', default='bart-base', type=str)
    parser.add_argument('--data_name', default='event_story', type=str)
    parser.add_argument("--resume_path", default='output/checkpoints/bart-base_event_story_ECoGM_2e-5', type=str)
    parser.add_argument('--resume_epoch', type=int, default=8)
    parser.add_argument("--log_info", type=str, default=None)
    parser.add_argument('--relation_index', type=int, default=None)

    parser.add_argument('--sampling_algorithm', default='beam-4', type=str)
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument("--history_length", type=int, default=100)

    parser.add_argument("--generate_explanation", action="store_true")

    parser.add_argument("--add_cls", action='store_true')

    parser.add_argument("--n_gpu", type=str, default='1', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--max_decode_len", default=20, type=int)

    args = parser.parse_args()

    init_logger(
        log_file=config['log_dir'] / f'{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}-{args.log_info}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    logger.info("Evaluation parameters %s", args)

    run_test(args)


if __name__ == '__main__':
    main()
