use attention::core::attn::*;
use candle_core::{DType, Device, Result, Tensor, D, IndexOp};
use candle_nn::{Dropout, Embedding, Linear, Module, ops::softmax};

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
    ln_f: DummyLayerNorm,
    out_head: Linear,
    device: Device,
}

impl DummyGptModel {
    pub fn new(config: GptConfig) -> Self {
        let device = get_device();
        let tok_emb_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.vocab_size, config.emb_dim),
            &device,
        )
        .unwrap();
        let tok_emb = Embedding::new(tok_emb_tensor, config.emb_dim);
        let pos_emb_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.context_length, config.emb_dim),
            &device,
        )
        .unwrap();
        let pos_emb = Embedding::new(pos_emb_tensor, config.emb_dim);
        let drop_emb = Dropout::new(config.drop_rate);
        let mut blocks = vec![];
        for _ in 0..config.n_layers {
            blocks.push(DummyTransformerBlock::new(config.clone()));
        }
        let ln_f = DummyLayerNorm::new(config.emb_dim, 1e-5);
        let out_head_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.vocab_size, config.emb_dim),
            &device,
        )
        .unwrap();
        let out_head = Linear::new(out_head_tensor, None);

        Self {
            tok_emb,
            pos_emb,
            drop_emb,
            blocks,
            ln_f,
            out_head,
            device,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, t) = x.shape().dims2()?;
        let tok_embeds = self.tok_emb.forward(x)?;
        let pos_embeds = self
            .pos_emb
            .forward(&Tensor::arange(0, t as u32, &self.device)?)?
            .broadcast_as(tok_embeds.shape())?;
        let x = tok_embeds.add(&pos_embeds)?;
        let mut x = self.drop_emb.forward(&x, false)?;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        // (batch_size, seq_len, emb_dim) * (emb_dim, vocab_size) -> (batch_size, seq_len, vocab_size)
        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}

pub struct DummyTransformerBlock {
    attn: MultiHeadAttention,
    ff: DummyFeedForward,
    norm1: DummyLayerNorm,
    norm2: DummyLayerNorm,
    dropout: Dropout,
}

impl DummyTransformerBlock {
    pub fn new(config: GptConfig) -> Self {
        let attn = MultiHeadAttention::new(
            config.emb_dim,
            config.emb_dim,
            config.context_length,
            config.n_heads,
            config.drop_rate,
            None,
            get_device(),
        );
        let ff = DummyFeedForward::new(config.clone(), None);
        let norm1 = DummyLayerNorm::new(config.emb_dim, 1e-5);
        let norm2 = DummyLayerNorm::new(config.emb_dim, 1e-5);
        let dropout = Dropout::new(config.drop_rate);
        Self {
            attn,
            ff,
            norm1,
            norm2,
            dropout,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let x = self.norm1.forward(&x)?;
        let x = self.attn.forward(&x)?;
        let x = self.dropout.forward(&x, false)?;
        let x = x.add(&shortcut)?;
        let x = self.norm2.forward(&x)?;
        let x = self.ff.forward(&x)?;
        let x = x.add(&shortcut)?;
        Ok(x)
    }
}

pub struct DummyLayerNorm {
    norm_eps: f64,
    weight: Tensor,
    bias: Tensor,
}

impl DummyLayerNorm {
    pub fn new(normalized_shape: usize, norm_eps: f64) -> Self {
        let device = get_device();
        let weight = Tensor::ones((normalized_shape,), DType::F32, &device).unwrap();
        let bias = Tensor::zeros((normalized_shape,), DType::F32, &device).unwrap();
        Self {
            norm_eps,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_mean = x.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&x_mean)?;
        let norm_x = x.var_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(norm_x + self.norm_eps)?.sqrt().unwrap())?;
        let x_normed = x_normed.to_dtype(DType::F32)?;
        let x_normed = x_normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?;
        Ok(x_normed)
    }
}

pub struct DummyFeedForward {
    linear_1: Linear,
    linear_2: Linear,
}

impl DummyFeedForward {
    pub fn new(config: GptConfig, bias: Option<Tensor>) -> Self {
        let device = get_device();
        let linear_1_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.emb_dim * 4, config.emb_dim),
            &device,
        )
        .unwrap();
        let linear_1 = Linear::new(linear_1_tensor, bias.clone());
        let linear_2_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.emb_dim, config.emb_dim * 4),
            &device,
        )
        .unwrap();
        let linear_2 = Linear::new(linear_2_tensor, bias.clone());
        Self { linear_1, linear_2 }
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

pub fn generate_text_simple(model: &DummyGptModel, idx: &Tensor, max_new_tokens: usize, context_size: usize) -> Result<Tensor> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let (b, t) = idx.shape().dims2()?;
        let start_idx = if t > context_size { t - context_size } else { 0 };
        let idx_cond = idx.i((.., start_idx..t))?;
        let logits = model.forward(&idx_cond)?;
        let (b, t, _) = logits.shape().dims3()?;
        let logits = logits.i((.., t - 1, ..))?;
        let probas = softmax(&logits, D::Minus1)?;
        let idx_next: Tensor = probas.argmax_keepdim(D::Minus1)?;
        idx = Tensor::cat(&[idx, idx_next], 1)?;
    }
    Ok(idx)
}