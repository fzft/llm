mod core;

pub use crate::core::gpt::*;

use candle_core::{Result, Tensor};
use tokenizers::Tokenizer;

pub fn stack_tensors() -> Result<Tensor> {
    let device = get_device();
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let text1 = "Every effort moves you";
    let text2 = "Every day holds a";
    let encoded1 = tokenizer.encode(text1, true).unwrap();
    let encoded2 = tokenizer.encode(text2, true).unwrap();
    let tensor1 =
        Tensor::from_vec(encoded1.get_ids().to_vec(), (1, encoded1.len()), &device).unwrap();
    let tensor2 =
        Tensor::from_vec(encoded2.get_ids().to_vec(), (1, encoded2.len()), &device).unwrap();
    let batch = Tensor::cat(&[tensor1, tensor2], 0).unwrap();
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, D};
    use candle_nn::{Linear, Module};

    #[test]
    fn it_works() {
        let batch = stack_tensors().unwrap();
        println!("batch: {}", batch);
    }

    #[test]
    pub fn test_gpt() {
        let batch = stack_tensors().unwrap();
        println!("batch shape: {:?}", batch.shape());
        let gpt = DummyGptModel::new(GptConfig {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_layers: 1,
            n_heads: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        });
        let output = gpt.forward(&batch).unwrap();
        println!("output shape: {}", output);
    }

    #[test]
    pub fn test_mean_var() {
        let x =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &get_device()).unwrap();
        let x_mean = x.mean_keepdim(D::Minus1).unwrap();
        let x = x.broadcast_sub(&x_mean).unwrap();
        let norm_x = x.var_keepdim(D::Minus1).unwrap();
        let x_normed = x.broadcast_div(&norm_x.sqrt().unwrap()).unwrap();
        println!("x_normed: {}", x_normed);
    }

    #[test]
    pub fn test_mean_var2() {
        let x =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &get_device()).unwrap();
        let hidden_size = x.dim(D::Minus1).unwrap();
        let mean_x = (x.sum_keepdim(D::Minus1).unwrap() / hidden_size as f64).unwrap();
        let x = x.broadcast_sub(&mean_x).unwrap();
        let norm_x = x.var_keepdim(D::Minus1).unwrap();
        let x_normed = x.broadcast_div(&norm_x.sqrt().unwrap()).unwrap();
        println!("x_normed: {}", x_normed);
    }

    #[test]
    pub fn test_broadcast_add() {
        let x = Tensor::from_vec(
            vec![1.0 as f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3, 2),
            &get_device(),
        )
        .unwrap();
        let y = 1.1;
        let z = (&x + y).unwrap();
        println!("z: {}", z);
    }

    #[test]
    pub fn test_gelu() {
        let x = Tensor::from_vec(
            vec![1.0 as f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3, 2),
            &get_device(),
        )
        .unwrap();
        let y = x.gelu().unwrap();
        println!("y: {}", y);
    }

    #[test]
    pub fn test_ffn() {
        let x = Tensor::rand(0.0f32, 1.0f32, (2, 4, 768), &get_device()).unwrap();
        let ffn = DummyFeedForward::new(
            GptConfig {
                vocab_size: 50257,
                context_length: 1024,
                emb_dim: 768,
                n_layers: 1,
                n_heads: 12,
                drop_rate: 0.1,
                qkv_bias: false,
            },
            None,
        );
        let y = ffn.forward(&x).unwrap();
        println!("y: {}", y);
    }

    #[test]
    pub fn test_linear() {
        let x = Tensor::rand(0.0f32, 1.0f32, (2, 4, 768), &get_device()).unwrap();
        let w = Tensor::rand(0.0f32, 1.0f32, (784 * 4, 768), &get_device()).unwrap();
        let linear = Linear::new(w, None);
        let y = linear.forward(&x).unwrap();
        println!("y: {}", y);
    }

    #[test]
    pub fn test_transformer_block() {
        let x = Tensor::rand(0.0f32, 1.0f32, (2, 4, 768), &get_device()).unwrap();
        let block = DummyTransformerBlock::new(GptConfig {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_layers: 1,
            n_heads: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        });
        let y = block.forward(&x).unwrap();
        println!("y: {}", y);
    }
}
