class CFG(object):
    def __init__(self, max_dec_length, tok, topK=None, eval_bs=None):
        self.max_dec_length = max_dec_length
        self.pad_token_id = tok.pad_token_id
        self.eos_token_id = tok.eos_token_id
        self.tok = tok
        self.topK = topK
        self.eval_bs = eval_bs

    def set_topK(self, topK):
        self.topK = topK

    def set_numbeams(self, num_beams):
        self.eval_bs = num_beams