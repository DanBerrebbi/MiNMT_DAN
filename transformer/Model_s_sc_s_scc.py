# -*- coding: utf-8 -*-

import sys
import os
import logging
import torch
import math
import numpy as np
import glob
from transformer.Model import Embedding, AddPositionalEncoding, Stacked_Encoder, Stacked_Decoder, Encoder, Decoder, Stacked_Encoder_scc, Stacked_Decoder_scc, Encoder_scc, Decoder_scc, MultiHead_Attn, MultiHead_Attn_Relu, FeedForward, Generator

##############################################################################################################
### Encoder_Decoder_s_sc_s_scc ######################################################################################
##############################################################################################################
class Encoder_Decoder_s_sc_s_scc(torch.nn.Module):
  #https://www.linzehui.me/images/16005200579239.jpg
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, share_embeddings, src_voc_size, tgt_voc_size, idx_pad):
    super(Encoder_Decoder_s_sc_s_scc, self).__init__()
    self.idx_pad = idx_pad
    self.src_emb = Embedding(src_voc_size, emb_dim, idx_pad) 
    self.tgt_emb = Embedding(tgt_voc_size, emb_dim, idx_pad) 
    
    self.layer_norm_1 = torch.nn.LayerNorm(emb_dim, eps=1e-6)
    self.layer_norm_2 = torch.nn.LayerNorm(emb_dim, eps=1e-6)

    self.multihead_attn_cross_pre = MultiHead_Attn_Relu(n_heads, emb_dim, qk_dim, v_dim, dropout)

    self.add_pos_enc = AddPositionalEncoding(emb_dim, dropout, max_len=5000) 
    self.stacked_encoder = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)         ### encoder for src and xsrc
    self.stacked_decoder = Stacked_Decoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)     ### decoder for tgt and xtgt

    self.generator_trn = Generator(emb_dim, tgt_voc_size)

  def type(self):
    return 's_sc_s_scc'

  def score_dan(self, msk_src, msk_xsrc):
    # msk_src is [bs, 1, l1] (False where <pad> True otherwise)
    # msk_xsrc is [bs, 1, l2] (False where <pad> True otherwise)
    alpha = []
    bs = msk_src.shape[0]
    for b in range(bs):
      lg_src, lg_xsrc = 0, 0
      for tok in msk_src[b][0]:
        if tok :
          lg_src+=1.
      for tok in msk_xsrc[b][0]:
        if tok :
          lg_xsrc+=1.
      score = abs(lg_src-lg_xsrc)/lg_src
      alpha.append(score)
    return torch.Tensor(alpha)


  def forward(self, src, xsrc, xtgt, tgt, msk_src, msk_xsrc, msk_xtgt_1, msk_xtgt_2, msk_tgt): 
    #src is [bs,ls]
    #tgt is [bs,lt]
    #msk_src is [bs,1,ls] (False where <pad> True otherwise)
    #mst_tgt is [bs,lt,lt]

    ## variable d'ecartement
    self.alpha = self.score_dan(msk_src, msk_xsrc)
    print(150*"££")
    print(self.alpha.shape)

    ### encoder #####
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder(xsrc, msk_xsrc) #[bs,ls,ed]
    
    ### decoder pre
    xtgt = self.add_pos_enc(self.tgt_emb(xtgt)) #[bs,ls,ed]
    z_xtgt = self.stacked_decoder.forward(z_xsrc, xtgt, msk_xsrc, msk_xtgt_1)

    ### decoder #####
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt = self.stacked_decoder.forward(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]
    
    z_tgt_pre = self.multihead_attn_cross_pre(q=z_tgt, k=z_xtgt, v=z_xtgt, msk=msk_xtgt_2)
    z_tgt = self.layer_norm_2(z_tgt + self.layer_norm_1(z_tgt_pre))
    ### generator ###
    y_tgt_trn = self.generator_trn(z_tgt) #[bs, lt, Vt]
    y_pre_trn = self.generator_trn(z_xtgt)
    return y_tgt_trn, y_pre_trn ### returns logits (for learning)

  def encode(self, src, xsrc, xtgt, msk_src, msk_xsrc, msk_xtgt_1):
    src = self.add_pos_enc(self.src_emb(src)) #[bs,ls,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    xsrc = self.add_pos_enc(self.src_emb(xsrc)) #[bs,ls,ed]
    z_xsrc = self.stacked_encoder(xsrc, msk_xsrc) #[bs,ls,ed]
    xtgt = self.add_pos_enc(self.tgt_emb(xtgt)) #[bs,ls,ed]
    z_xtgt = self.stacked_decoder.forward(z_xsrc, xtgt, msk_xsrc, msk_xtgt_1) #[bs,ls,ed]
    return z_src, z_xtgt

  def decode(self, z_src, z_xtgt, tgt, msk_src, msk_xtgt_2, msk_tgt=None):
    assert z_src.shape[0] == tgt.shape[0] ### src/tgt batch_sizes must be equal
    #z_src are the embeddings of the source words (encoder) [bs, sl, ed]
    #tgt is the history (words already generated) for current step [bs, lt]
    tgt = self.add_pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_tgt =  self.stacked_decoder.forward(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]
    z_tgt_pre = self.multihead_attn_cross_pre(q=z_tgt, k=z_xtgt, v=z_xtgt, msk=msk_xtgt_2)
    z_tgt = self.layer_norm_2(z_tgt + self.layer_norm_1(z_tgt_pre))
    ### generator ###
    y = self.generator_trn(z_tgt) #[bs, lt, Vt]
    y = torch.nn.functional.log_softmax(y, dim=-1) 
    return y ### returns log_probs (for inference)



