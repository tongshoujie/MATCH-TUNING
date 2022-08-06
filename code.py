# x : the original PLM outputs
# for the current batch, shape=[bsz, hid]
# self.dense : hid to hid dense

# match : the match matrix
# for the current batch, shape=[bsz, bsz]

def MatchTuning(self, outputs):
    ori_rep = outputs[0][:, 0]    # bert cls
    ori_rep = self.dense(ori_rep)
    rev_rep = ori_rep.transpose(-1, -2)
    match = torch.matmul(ori_rep, rev_rep) / temperature
    match = nn.Softmax(dim=-1)(match)
    match_rep = torch.matmul(match, ori_rep)

    return match_rep