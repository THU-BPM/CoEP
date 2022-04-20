import torch
import time
from argparse import ArgumentParser
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from src.common.tools import init_logger, logger
from configs.basic_config import config

from src.test.sampler import GreedySampler, TopKSampler, BeamSampler
from src.test.sampler_config import CFG
from configs.finetuning_config import config as ft_config
from src.data.bart_processor_interactive import create_dense_feature

from src.test.generator_bart_interactive import Generator

from src.model.bart_for_generation import BartWithClassification


def set_sampler(cfg, sampling_algorithm):
    if "beam" in sampling_algorithm:
        cfg.set_numbeams(int(sampling_algorithm.split("-")[1]))
        sampler = BeamSampler(cfg)
    elif "topk" in sampling_algorithm:
        cfg.set_topK(int(sampling_algorithm.split("-")[1]))
        sampler = TopKSampler(cfg)
    else:
        sampler = GreedySampler(cfg)

    return sampler


def preprocess_test_data(sentences, args, tokenizer):
    from src.data.utils import collate_fn_test
    from src.data.bart_processor_interactive import create_test_dataset

    test_dataset = create_test_dataset(sentences, tokenizer, args.max_seq_len, args.history_length)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size,
                                 collate_fn=collate_fn_test)
    return test_dataloader


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bart-base', type=str)
    parser.add_argument('--logger_info', default='', type=str)
    parser.add_argument('--filename', default='dataset/event_story/event_story_test_refs.csv', type=str)
    parser.add_argument('--output_file', default='generate', type=str)
    parser.add_argument('--model_path', default=None, type=str) # EP, EP_clari

    parser.add_argument('--add_cls', action='store_true')

    parser.add_argument('--sampling_algorithm', default='beam-4', type=str)
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')

    parser.add_argument("--history_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--max_decode_len", default=20, type=int)

    args = parser.parse_args()

    init_logger(
        log_file=f"{config['log_dir']}/{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}-{args.logger_info}.log")
    logger.info("Generation parameters %s", args)

    if args.add_cls:
        from src.model.bart_for_generation import BartWithClassification
        model = BartWithClassification(path=str(config[f'{args.arch}_model_dir']), num_labels=2)
    else:
        model = BartForConditionalGeneration.from_pretrained(str(config[f'{args.arch}_model_dir']))

    model.load_state_dict(torch.load(ft_config[args.model_path], map_location=f"cuda:{args.n_gpu}")['model'])

    tok = BartTokenizer.from_pretrained(str(config[f'{args.arch}_model_dir']))
    cfg = CFG(args.max_decode_len, tok)
    sampler = set_sampler(cfg, args.sampling_algorithm)
    generator = Generator(model=model, tok=tok, logger=logger, n_gpu=args.n_gpu)
    num_return_sequences = 1

    if args.filename is None:
        while input("stop generating? ") != "yes":
            context = input("context: ")
            event = input("event: ")

            data = create_dense_feature({'context': context, 'event': event}, tokenizer=tok,
                                        max_seq_len=args.max_seq_len,
                                        his_len=args.history_length)
            # ----------- predicting ----------
            logger.info('interactive generating....')

            result_data, input_sent, seqs = generator.interactive_generate(data=data)

            logger.info(f"input: {input_sent}, generated: {seqs}")
    else:
        import pandas as pd
        stories = pd.read_csv(args.filename)
        sentences = [[story.split('\t')[1]] for story in stories['context'].values]

        test_dataloader = preprocess_test_data(sentences, args=args, tokenizer=tok)

        for i in range(4):
            all_generated_sents = generator.generate(
                data=test_dataloader,
                max_length=args.max_decode_len,
                length_penalty=args.length_penalty,
                num_beams=4,
                num_return_sequences=num_return_sequences,
                early_stopping=True)
            logger.info(f"type all_generated_sents: {type(all_generated_sents)}\nsample: {all_generated_sents[0:10]}")
            assert len(sentences) == len(all_generated_sents)

            for i in range(len(sentences)):
                sentences[i].append(str(all_generated_sents[i]))
            # sentences = [list(s).extend([str(g)]) for s, g in zip(sentences, all_generated_sents)]
            logger.info(f"type sentences: {type(sentences)}\n type sentence:{type(sentences[0])}")
            test_dataloader = preprocess_test_data(sentences=sentences, args=args, tokenizer=tok)

        df = pd.DataFrame(columns=["generated"])
        df["generated"] = [' '.join(s) for s in sentences]
        df.to_csv(f'{args.output_file}.csv')


if __name__ == '__main__':
    main()
