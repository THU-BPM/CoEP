from pathlib import Path
BASE_DIR = Path('../ECoGM')

config = {
    'data_dir':  BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'output/log',
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'result_dir':  BASE_DIR / "output/result",

    'bart-base_model_dir':  BASE_DIR / 'pretrain/BART/facebook/bart-base',
    'bart-base_vocab_path':  BASE_DIR / 'pretrain/BART/facebook/bart-base/vocab.jason',
    'bart-base_merge_path':  BASE_DIR / 'pretrain/BART/facebook/bart-base/merges.txt',
    'bart-base_config_path': BASE_DIR / 'pretrain/BART/facebook/bart-base/config.json',

    'bart-large_model_dir':  BASE_DIR / 'pretrain/BART/facebook/bart-large',
    'bart-large_vocab_path': BASE_DIR / 'pretrain/BART/facebook/bart-large/vocab.jason',
    'bart-large_merge_path': BASE_DIR / 'pretrain/BART/facebook/bart-large/merges.txt',
    'bart-large_config_path':  BASE_DIR / 'pretrain/BART/facebook/bart-large/config.json',

    "relations": ["PersonX Need",
                  "PersonX Intent",
                  "PersonX Attr",
                  "PersonX Effect",
                  "PersonX React",
                  "PersonX Want",
                  "Other Effect",
                  "Other React",
                  "Other Want"
                  ]
}
