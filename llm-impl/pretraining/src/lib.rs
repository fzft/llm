use candle_core::{Tensor, D, Result};
use tokenizers::Tokenizer;  
use gpt::get_device;

pub fn text_to_token_ids(text: &str) -> Result<Tensor> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let tokens = tokenizer.encode(text, true).unwrap();
    let idx = Tensor::from_vec(tokens.get_ids().to_vec(), (1, tokens.len()), &get_device()).unwrap();
    Ok(idx)
}

pub fn token_ids_to_text(token_ids: &Tensor) -> Result<String> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let text = tokenizer.decode(&token_ids.squeeze(0)?.to_vec1()?, true).unwrap();
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_token_ids() {
        let text = "Hello, I am a";
        let token_ids = text_to_token_ids(text).unwrap();
        println!("token_ids: {:?}", token_ids);
    }

    #[test]
    fn test_token_ids_to_text() {
        let text = "Hello, I am a";
        let token_ids = text_to_token_ids(text).unwrap();
        let text = token_ids_to_text(&token_ids).unwrap();
        assert_eq!(text, "Hello, I am a");
    }


}
