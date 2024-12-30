use std::ops::Index;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer as TiktokenTokenizer;

#[derive(Clone)]
pub struct Dataset {
    pub data: Vec<(Tensor, Tensor)>,
}

impl Dataset {
    pub fn new(
        txt: &str,
        tokenizer: &TiktokenTokenizer,
        max_seq_len: usize,
        stride: usize,
    ) -> Self {
        let device = Device::Cpu;
        let tokens = tokenizer.encode(txt, true).unwrap();
        let mut data = Vec::<(Tensor, Tensor)>::new();
        for i in (0..tokens.len() - max_seq_len).step_by(stride) {
            let input_chunk = tokens.get_ids()[i..i + max_seq_len].to_vec();
            let target_chunk = tokens.get_ids()[i + 1..i + max_seq_len + 1].to_vec();
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

impl Iterator for Dataset {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        self.data.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;
    use candle_datasets::Batcher;

    #[test]
    fn test_dataset() {
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let dataset = Dataset::new(&text_data, &tokenizer, 1024, 1024);
        println!("dataset: {:?}", dataset.len());
    }

    #[test]
    fn test_iterator() {
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let dataset = Dataset::new(&text_data, &tokenizer, 1024, 1024);
        for (input, target) in dataset {
            println!("input: {:?}", input);
            println!("target: {:?}", target);
        }
    }

    #[test]
    fn test_dataloader() {
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let dataset = Dataset::new(&text_data, &tokenizer, 4, 1);
        let dataloader = Batcher::new2(dataset).batch_size(1);
        for batch in dataloader {
            println!("batch: {:?}", batch);
        }
    }
}
