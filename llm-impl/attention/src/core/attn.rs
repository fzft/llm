use core::f64;

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{Dropout, Linear, Module};

pub struct MultiHeadAttention {
    device: Device,
    d_out: usize,
    n_heads: usize,
    dropout: Dropout,
    bias: Option<Tensor>,
    d_head: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    out_proj: Linear,
    mask: Tensor,
}

impl MultiHeadAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        t: usize,
        n_heads: usize,
        dropout: f32,
        bias: Option<Tensor>,
    ) -> Self {
        assert!(d_out % n_heads == 0, "d_out must be divisible by n_heads");
        let d_head = d_out / n_heads;
        let device = Device::Cpu;
        let dropout = Dropout::new(dropout);
        let w_q = Linear::new(
            Tensor::rand(0.0, 1.0, (d_out, d_in), &device).unwrap(),
            bias.clone(),
        );
        let w_k = Linear::new(
            Tensor::rand(0.0, 1.0, (d_out, d_in), &device).unwrap(),
            bias.clone(),
        );
        let w_v = Linear::new(
            Tensor::rand(0.0, 1.0, (d_out, d_in), &device).unwrap(),
            bias.clone(),
        );

        let out_proj = Linear::new(
            Tensor::rand(0.0, 1.0, (d_out, d_out), &device).unwrap(),
            bias.clone(),
        );

        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &device).unwrap();
        Self {
            device,
            d_out,
            n_heads,
            dropout,
            bias,
            d_head,
            w_q,
            w_k,
            w_v,
            out_proj,
            mask,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d_in) = x.shape().dims3()?;
        println!("b: {}, t: {}, d_in: {}", b, t, d_in);
        // (b, t, d_in) -> (b, t, d_out)
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // (b, t, d_out) -> (b, t, n_heads, d_head)
        let q = q.reshape((b, t, self.n_heads, self.d_head))?;
        let k = k.reshape((b, t, self.n_heads, self.d_head))?;
        let v = v.reshape((b, t, self.n_heads, self.d_head))?;

        // (b, t, n_heads, d_head) -> (b, n_heads, t, d_head)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // (b, n_heads, t, d_head) -> (b, n_heads, t, t)
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let mask = self.mask.i((0..t, 0..t))?;
        let mask = mask.broadcast_as(attn.shape())?;
        let attn = masked_fill(&attn, &mask, f64::NEG_INFINITY)?;
        let attn = softmax(&scale_scores(attn)?, D::Minus1)?;
        let context_vec = attn.matmul(&v)?.transpose(1, 2)?;
        // (b, n_heads, t, d_head) -> (b, t, d_out)
        let context_vec = context_vec.contiguous()?.reshape((b, t, self.d_out))?;
        println!("context_vec: {}", context_vec);
        // (b, t, d_out) -> (b, t, d_out)
        let out = self.out_proj.forward(&context_vec)?;
        Ok(out)
    }
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f64) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, &on_false)?;
    Ok(m)
}

fn scale_scores(scores: Tensor) -> Result<Tensor> {
    let device = scores.device();
    let (_, _, _, dk) = scores.shape().dims4()?;
    let scale_factor = Tensor::new(vec![dk as f64], &device)?.sqrt()?;
    scores.broadcast_div(&scale_factor)
}
