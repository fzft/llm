use crate::core::utils::preprocess;
use std::collections::HashMap;
use tokenizers::tokenizer::Tokenizer as TiktokenTokenizer;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}

pub struct SimpleTokenizerV1 {
    pub str_to_int: HashMap<String, u32>,
    pub int_to_str: HashMap<u32, String>,
}

impl SimpleTokenizerV1 {
    pub fn new(vocab_map: HashMap<String, u32>) -> Self {
        Self {
            str_to_int: vocab_map.clone(),
            int_to_str: vocab_map.into_iter().map(|(k, v)| (v, k)).collect(),
        }
    }
}

impl Tokenizer for SimpleTokenizerV1 {
    fn encode(&self, text: &str) -> Vec<u32> {
        let preprocessed = preprocess(text);
        println!("preprocessed: {:?}", preprocessed);
        let preprocessed: Vec<String> = preprocessed
            .into_iter()
            .map(|word| {
                if self.str_to_int.contains_key(&word) {
                    word
                } else {
                    "<unk>".to_string()
                }
            })
            .collect();
        preprocessed
            .iter()
            .map(|word| *self.str_to_int.get(word).unwrap_or(&u32::MIN))
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&i| self.int_to_str.get(&i))
            .cloned()
            .collect::<Vec<String>>()
            .join(" ")
    }
}

pub struct SimpleTokenizerV2 {
    pub str_to_int: HashMap<String, u32>,
    pub int_to_str: HashMap<u32, String>,
}

impl SimpleTokenizerV2 {
    pub fn new(vocab_map: HashMap<String, u32>) -> Self {
        Self {
            str_to_int: vocab_map.clone(),
            int_to_str: vocab_map.into_iter().map(|(k, v)| (v, k)).collect(),
        }
    }
}

impl Tokenizer for SimpleTokenizerV2 {
    fn encode(&self, text: &str) -> Vec<u32> {
        let tokenizer = TiktokenTokenizer::from_pretrained("gpt2", None).unwrap();
        let encoded = tokenizer.encode(text, true).unwrap();
        encoded.get_ids().to_vec()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let tokenizer = TiktokenTokenizer::from_pretrained("gpt2", None).unwrap();
        let decoded = tokenizer.decode(tokens, true).unwrap();
        decoded
    }
}
