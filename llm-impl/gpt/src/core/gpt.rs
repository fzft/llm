use std::collections::HashMap;

use attention::core::attn::*;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, ops::softmax, Dropout, Embedding, Init, LayerNorm, Linear,
    Module, VarBuilder, VarMap,
};

#[derive(Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

pub struct DummyGptModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    blocks: Vec<DummyTransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

/*
DummyGptModel VarBuilder Architecture
tok_emb: (vocab_size, emb_dim)
pos_emb: (context_length, emb_dim)
drop_emb: (context_length, emb_dim)
blocks: (n_layers, emb_dim, emb_dim)
ln_f: (context_length, emb_dim)
out_head: (emb_dim, vocab_size)
*/

impl DummyGptModel {
    pub fn new(vb: VarBuilder, config: GptConfig) -> Result<Self> {
        // let mut map = HashMap::new();
        // map.insert(
        //     String::from("weight"),
        //     Tensor::randn(0f32, 1., (config.vocab_size, config.emb_dim), &device)?,
        // );
        // let vb = VarBuilder::from_tensors(map, DType::F32, &device);
        // wte: (vocab_size, emb_dim)
        let transformer = vb.pp("transformer");
        let tok_emb = embedding(config.vocab_size, config.emb_dim, transformer.pp("wte"))?;
        // let mut map2 = HashMap::new();
        // map2.insert(
        //     String::from("weight"),
        //     Tensor::rand(0f32, 1., (config.context_length, config.emb_dim), &device)?,
        // );
        // let vb2 = VarBuilder::from_tensors(map2, DType::F32, &device);
        // wpe: (context_length, emb_dim)
        let pos_emb = embedding(config.context_length, config.emb_dim, transformer.pp("wpe"))?;
        let drop_emb = Dropout::new(config.drop_rate);
        let mut blocks = vec![];
        for i in 0..config.n_layers {
            blocks.push(DummyTransformerBlock::new(
                transformer.pp(format!("h.{i}")),
                config.clone(),
            )?);
        }
        // let vmap = VarMap::new();
        // let vb = VarBuilder::from_varmap(&vmap, DType::F32, &device);
        let ln_f = layer_norm(config.emb_dim, 1e-5, transformer.pp("ln_f"))?;
        let lm_head = linear(config.emb_dim, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            tok_emb,
            pos_emb,
            drop_emb,
            blocks,
            ln_f,
            lm_head,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, t) = x.shape().dims2()?;
        let tok_embeds = self.tok_emb.forward(x)?;
        let pos_embeds = self
            .pos_emb
            .forward(&Tensor::arange(0, t as u32, x.device())?)?
            .broadcast_as(tok_embeds.shape())?;
        let x = tok_embeds.add(&pos_embeds)?;
        let mut x = self.drop_emb.forward(&x, false)?;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        // (batch_size, seq_len, emb_dim) * (emb_dim, vocab_size) -> (batch_size, seq_len, vocab_size)
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
}

pub struct DummyTransformerBlock {
    attn: MultiHeadAttention,
    ff: DummyFeedForward,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    dropout: Dropout,
}

impl DummyTransformerBlock {
    pub fn new(vb: VarBuilder, config: GptConfig) -> Result<Self> {
        let attn = MultiHeadAttention::new(
            vb.pp("attn"),
            config.emb_dim,
            config.emb_dim,
            config.context_length,
            config.n_heads,
        )?;
        let ff = DummyFeedForward::new(vb.pp("mlp"), config.clone())?;
        let ln_1 = layer_norm(config.emb_dim, 1e-5, vb.pp("ln_1"))?;
        let ln_2 = layer_norm(config.emb_dim, 1e-5, vb.pp("ln_2"))?;
        let dropout = Dropout::new(config.drop_rate);
        Ok(Self {
            attn,
            ff,
            ln_1,
            ln_2,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.ln_1.forward(&x)?;
        let x = self.attn.forward(&x)?;
        let x = self.dropout.forward(&x, false)?;
        let x = x.add(&shortcut)?;
        let x = self.ln_2.forward(&x)?;
        let x = self.ff.forward(&x)?;
        let x = x.add(&shortcut)?;
        Ok(x)
    }
}

pub struct DummyFeedForward {
    linear_1: Linear,
    linear_2: Linear,
}

impl DummyFeedForward {
    pub fn new(vb: VarBuilder, config: GptConfig) -> Result<Self> {
        let linear_1 = linear(config.emb_dim, config.emb_dim * 4, vb.pp("c_fc"))?;
        let linear_2 = linear(config.emb_dim * 4, config.emb_dim, vb.pp("c_proj"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (batch_size, seq_len, emb_dim) * (emb_dim, emb_dim * 4) -> (batch_size, seq_len, emb_dim * 4)
        let x = self.linear_1.forward(&x)?;
        // (batch_size, seq_len, emb_dim * 4) -> (batch_size, seq_len, emb_dim * 4)
        let x = x.gelu()?;
        // (batch_size, seq_len, emb_dim * 4) * (emb_dim * 4, emb_dim) -> (batch_size, seq_len, emb_dim)
        let x = self.linear_2.forward(&x)?;
        Ok(x)
    }
}

pub fn get_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap()
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    }
}

pub fn generate_text_simple(
    model: &DummyGptModel,
    idx: &Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let (b, t) = idx.shape().dims2()?;
        let start_idx = if t > context_size {
            t - context_size
        } else {
            0
        };
        let idx_cond = idx.i((.., start_idx..t))?;
        let logits = model.forward(&idx_cond)?;
        let (b, t, _) = logits.shape().dims3()?;
        let logits = logits.i((.., t - 1, ..))?;
        let probas = softmax(&logits, D::Minus1)?;
        dbg!(&probas);
        let idx_next: Tensor = probas.argmax_keepdim(D::Minus1)?;
        idx = Tensor::cat(&[idx, idx_next], 1)?;
    }
    Ok(idx)
}
