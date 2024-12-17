use working_with_text_data::{
    download_file, get_vocab, preprocess, read_file, resplit, SimpleTokenizerV1, SimpleTokenizerV2, text_join, Tokenizer as TokenizerV1
};

use tokenizers::tokenizer::Tokenizer;

#[cfg(test)]
mod tokenize_test {
    use std::collections::HashSet;

    use super::*;

    #[tokio::test]
    async fn download_file_test() {
        let url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt";
        let file_path = "the-verdict.txt";
        download_file(url, file_path).await.unwrap();
    }

    #[tokio::test]
    async fn read_file_test() {
        let file_path = "the-verdict.txt";
        let content = read_file(file_path).await.unwrap();
        assert!(!content.is_empty());
        println!("Total number of characters: {}", content.len());
    }

    #[tokio::test]
    async fn resplit_test() {
        let text = "Hello, world. This, is a test.";
        let pattern = r"(\s+|\S+)";
        let tokens = resplit(text, pattern);
        println!("{:?}", tokens);
    }

    #[tokio::test]
    async fn resplit_test_2() {
        let text = "Hello, world. Is() this-- a \"test?";
        let pattern = r#"(\w+|[,.:;?_!'"()--])"#;
        let tokens = resplit(text, pattern);
        println!("{:?}", tokens);
    }

    #[tokio::test]
    async fn preprocess_test() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let tokens = preprocess(&content);
        println!("length: {}", tokens.len());
    }

    #[tokio::test]
    async fn preprocess_test_2() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let preprocessed = preprocess(&content);
        let uniq_tokens: HashSet<String> = preprocessed.iter().cloned().collect();
        println!("Total number of unique tokens: {}", uniq_tokens.len());
        let mut all_words: Vec<String> = uniq_tokens.into_iter().collect();
        all_words.sort();

        let vocab = all_words
            .iter()
            .enumerate()
            .map(|(i, word)| format!("{:4}: {}", i, word))
            .collect::<Vec<String>>();

        // print the first 100 words
        println!("{}", vocab[..100].join("\n"));
    }

    #[tokio::test]
    async fn simple_tokenizer_v1_encode_test() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let vocab = get_vocab(&content);
        let simple_tokenizer = SimpleTokenizerV1::new(vocab);
        let encoded = simple_tokenizer.encode("It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.");
        println!("{:?}", encoded);
    }

    #[tokio::test]
    async fn simple_tokenizer_v1_encode_test_2() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let vocab = get_vocab(&content);
        let simple_tokenizer = SimpleTokenizerV1::new(vocab);
        let encoded = simple_tokenizer.encode("I'm not sure I understand you.");
        println!("{:?}", encoded);
    }

    #[tokio::test]
    async fn simple_tokenizer_v1_decode_test() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let vocab = get_vocab(&content);
        let simple_tokenizer = SimpleTokenizerV1::new(vocab);
        let encoded = simple_tokenizer.encode("It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.");
        let decoded = simple_tokenizer.decode(&encoded);
        println!("{}", decoded);
    }

    #[tokio::test]
    async fn simple_tokenizer_v1_text_join_test() {
        let texts = ["Hello, do you like tea?", "In the sunlit terraces of the palace."];
        let joined = text_join(&texts);
        println!("{}", joined);
        let content = read_file("the-verdict.txt").await.unwrap();
        let vocab = get_vocab(&content);
        let simple_tokenizer = SimpleTokenizerV1::new(vocab);
        let encoded = simple_tokenizer.encode(&joined);
        println!("{:?}", encoded);

        println!("{}", simple_tokenizer.decode(&encoded));
    }

    #[tokio::test]
    async fn simple_tokenizer_v1_encode_with_tiktoken_test() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let vocab = get_vocab(&content);
        let simple_tokenizer = SimpleTokenizerV2::new(vocab);
        let encoded = simple_tokenizer.encode("It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.");
        println!("{:?}", encoded);
    }

    #[tokio::test]
    async fn simple_tokenizer_v1_decode_with_tiktoken_test() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let vocab = get_vocab(&content);
        let simple_tokenizer = SimpleTokenizerV2::new(vocab);
        let encoded = simple_tokenizer.encode("It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.");
        let decoded = simple_tokenizer.decode(&encoded);
        println!("{}", decoded);
    }

    #[tokio::test]
    async fn preprocess_2_test() {
        let content = read_file("the-verdict.txt").await.unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let tokens = tokenizer.encode(content.as_str(), true).unwrap();
        println!("length: {}", tokens.len());
        let enc_sample = tokens.get_ids()[..50].to_vec();
        let content_size = 4;
        let x = enc_sample[..content_size].to_vec();
        let y = enc_sample[1..content_size + 1].to_vec();
        for i in 1..(content_size + 1) {
            let context = x[..i].to_vec();
            let target = y[i - 1];
            println!("{:?}, ---> {:?}", context, target);
        }

        for i in 1..(content_size + 1) {
            let context = x[..i].to_vec();
            let target = y[i - 1];
            println!("{:?}, ---> {:?}", tokenizer.decode(&context, true).unwrap(), tokenizer.decode(&[target], true).unwrap());
        }
    }
}
