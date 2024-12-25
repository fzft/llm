
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{linear, Dropout, Linear, Module, VarBuilder, VarMap};

pub struct MultiHeadAttention {
    d_out: usize,
    n_heads: usize,
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
        device: Device,
    ) -> Result<Self> {
        assert!(d_out % n_heads == 0, "d_out must be divisible by n_heads");
        let vmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&vmap, DType::F32, &device);
        let d_head = d_out / n_heads;
        let w_q = linear(d_out, d_in, vb.pp("w_q"))?;
        let w_k = linear(d_out, d_in, vb.pp("w_k"))?;
        let w_v = linear(d_out, d_in, vb.pp("w_v"))?;
        let out_proj = linear(d_out, d_out, vb.pp("out_proj"))?;

        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &device).unwrap();
        Ok(Self {
            d_out,
            n_heads,
            d_head,
            w_q,
            w_k,
            w_v,
            out_proj,
            mask,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d_in) = x.shape().dims3()?;
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
        let attn = masked_fill(&attn, &mask, f32::NEG_INFINITY)?;
        let attn = softmax(&scale_scores(attn)?, D::Minus1)?;
        let context_vec = attn.matmul(&v)?.transpose(1, 2)?;
        // (b, n_heads, t, d_head) -> (b, t, d_out)
        let context_vec = context_vec.contiguous()?.reshape((b, t, self.d_out))?;
        // (b, t, d_out) -> (b, t, d_out)
        let out = self.out_proj.forward(&context_vec)?;
        Ok(out)
    }
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, &on_false)?;
    Ok(m)
}

fn scale_scores(scores: Tensor) -> Result<Tensor> {
    let device = scores.device();
    let (_, _, _, dk) = scores.shape().dims4()?;
    let scale_factor = Tensor::new(vec![dk as f32], &device)?.sqrt()?;
    scores.broadcast_div(&scale_factor)
}
