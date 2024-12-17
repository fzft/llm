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
    use candle_core::{Device, Tensor};

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
        println!("output shape: {:?}", output.shape());
    }
}
