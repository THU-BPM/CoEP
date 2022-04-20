from pathlib import Path

BASE_DIR = Path('output/checkpoints')
config = {
    'atomic_kg_model': BASE_DIR / 'bart-base_v4_atomic_KG_lr2e-5/epoch_4_bart-base_model.bin',
    'atomic_kg_wcls_model': BASE_DIR / 'bart-base_v4_atomic_KG_wcls/epoch_4_bart-base_model.bin',
    'ECoGM': BASE_DIR/ 'bart-base_event_story_ECoGM/epoch_4_bart-base_model.bin',
    'conceptnet_seq_model': BASE_DIR / 'bart-base_conceptnet_seq_GM/epoch_4_bart-base_model.bin',
    'EP_clari': BASE_DIR / 'bart-base_event_story_EP_kg_clari/epoch_7_bart-base_model.bin',
    'EP': BASE_DIR/ 'bart-base_event_story_EP/epoch_4_bart-base_model.bin'
}
