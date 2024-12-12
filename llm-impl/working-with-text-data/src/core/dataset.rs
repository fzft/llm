use std::ops::Index;

use crate::core::tokenize::Tokenizer;
use candle_core::{Device, Tensor};

struct Dataset {
    pub data: Vec<(Tensor, Tensor)>,
}

impl Dataset {
    pub fn new(txt: &str, tokenizer: &impl Tokenizer, max_seq_len: usize, stride: usize) -> Self {
        let device = Device::Cpu;
        let tokens = tokenizer.encode(txt);
        let mut data = Vec::<(Tensor, Tensor)>::new();
        for i in (0..tokens.len() - max_seq_len).step_by(stride) {
            let input_chunk = tokens[i..i + max_seq_len].to_vec();
            let target_chunk = tokens[i + 1..i + max_seq_len + 1].to_vec();
            let input_tensor = Tensor::new(input_chunk, &device).unwrap();
            let target_tensor: Tensor = Tensor::new(target_chunk, &device).unwrap();
            data.push((input_tensor, target_tensor));
        }
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Index<usize> for Dataset {
    type Output = (Tensor, Tensor);

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}