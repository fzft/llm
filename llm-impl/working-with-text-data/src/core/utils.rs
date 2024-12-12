use std::collections::{HashMap, HashSet};

use futures_util::StreamExt;
use regex::Regex;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub async fn download_file(url: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    let mut file = File::create(file_path).await?;
    let mut stream = response.bytes_stream();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        file.write_all(&chunk).await?;
    }
    Ok(())
}

pub async fn read_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(file_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut content = String::new();
    while let Some(line) = lines.next_line().await? {
        content.push_str(&line);
    }
    Ok(content)
}

pub fn resplit(text: &str, pattern: &str) -> Vec<String> {
    let re = Regex::new(pattern).unwrap();
    re.find_iter(text).map(|s| s.as_str().to_string()).collect()
}

pub fn preprocess(content: &str) -> Vec<String> {
    let tokens = resplit(content, r#"(\w+|[,.:;?_!"()'--]|")"#);
    tokens.iter().map(|s| s.to_string()).collect()
}

pub fn get_vocab(content: &str) -> HashMap<String, u32> {
    let preprocessed = preprocess(content);
    let uniq_tokens: HashSet<String> = preprocessed.iter().cloned().collect();
    let mut all_words: Vec<String> = uniq_tokens.into_iter().collect();
    all_words.sort();
    all_words.push("<|endoftext|>".to_string());
    all_words.push("<unk>".to_string());
    all_words.iter().cloned().enumerate().map(|(i, word)| (word, i as u32)).collect()
}

pub fn text_join(texts: &[&str]) -> String {
    texts.join(" <|endoftext|> ")
}

