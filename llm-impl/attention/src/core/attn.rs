use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct MultiHeadAttention {
    d_out: usize,
    n_heads: usize,
    d_head: usize,
    qkv_linear: Linear,
    out_proj: Linear,
    mask: Tensor,
}

impl MultiHeadAttention {
    pub fn new(
        vb: VarBuilder,
        d_in: usize,
        d_out: usize,
        t: usize,
        n_heads: usize,
    ) -> Result<Self> {
        assert!(d_out % n_heads == 0, "d_out must be divisible by n_heads");
        let d_head = d_out / n_heads;
        let qkv_linear = linear(d_in, d_out * 3, vb.pp("c_attn"))?;
        // let w_q = linear(d_out, d_in, vb.pp("w_q"))?;
        // let w_k = linear(d_out, d_in, vb.pp("w_k"))?;
        // let w_v = linear(d_out, d_in, vb.pp("w_v"))?;
        let out_proj = linear(d_out, d_out, vb.pp("c_proj"))?;

        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), vb.device()).unwrap();
        Ok(Self {
            d_out,
            n_heads,
            d_head,
            qkv_linear,
            out_proj,
            mask,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d_in) = x.shape().dims3()?;
        // (b, t, d_in) -> (b, t, d_out)
        let qkv = self.qkv_linear.forward(x)?;
        // (b, t, d_out) -> (b, t, n_heads, d_head)
        let qkv = qkv.reshape((b, t, self.n_heads, self.d_head))?;
        let q = qkv.i((0, .., 0, ..))?;
        let k = qkv.i((0, .., 1, ..))?;
        let v = qkv.i((0, .., 2, ..))?;

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
