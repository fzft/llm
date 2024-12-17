use candle_core::{Device, Result, Shape, Tensor};
use candle_nn::{Dropout, Embedding, LayerNorm, Linear, Module};

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
        println!("device: {:?}", device);
        let tok_emb_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.vocab_size, config.emb_dim),
            &device,
        )
        .unwrap();
        println!("tok_emb_tensor shape: {:?}", tok_emb_tensor.shape());
        let tok_emb = Embedding::new(tok_emb_tensor, config.emb_dim);
        let pos_emb_tensor = Tensor::rand(
            0.0 as f32,
            1.0 as f32,
            (config.context_length, config.emb_dim),
            &device,
        )
        .unwrap();
        println!("pos_emb_tensor shape: {:?}", pos_emb_tensor.shape());
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
            (config.emb_dim, config.vocab_size),
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
        println!("x shape: {:?}", x.shape());
        // (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        let tok_embeds = self.tok_emb.forward(x)?;
        println!("tok_embeds shape: {:?}", tok_embeds.shape());
        // (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        let pos_embeds = self
            .pos_emb
            .forward(&Tensor::arange(0, t as u32, &self.device)?)?
            .broadcast_as(tok_embeds.shape())?;
        println!("pos_embeds shape: {:?}", pos_embeds.shape());
        let x = tok_embeds.add(&pos_embeds)?;
        let mut x = self.drop_emb.forward(&x, false)?;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}

pub struct DummyTransformerBlock {
    config: GptConfig,
}

impl DummyTransformerBlock {
    pub fn new(config: GptConfig) -> Self {
        Self { config }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

pub struct DummyLayerNorm {}

impl DummyLayerNorm {
    pub fn new(normalized_shape: usize, norm_eps: f32) -> Self {
        Self {}
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
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
