use candle_core::{IndexOp, Result, Tensor, D};
use gpt::get_device;
use tokenizers::Tokenizer;

pub fn text_to_token_ids(text: &str) -> Result<Tensor> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let tokens = tokenizer.encode(text, false).unwrap();
    let idx =
        Tensor::from_vec(tokens.get_ids().to_vec(), (1, tokens.len()), &get_device()).unwrap();
    Ok(idx)
}

pub fn token_ids_to_text(token_ids: &Tensor) -> Result<String> {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let text = tokenizer
        .decode(&token_ids.squeeze(0)?.to_vec1()?, false)
        .unwrap();
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use candle_nn::ops::softmax;
    use candle_nn::loss::cross_entropy;
    use gpt::*;

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

    #[test]
    fn test_generate_text(){
        let inputs: Tensor = Tensor::from_vec(
            vec![
                16833 as u32,
                3626 as u32,
                6100 as u32,
                40 as u32,
                1107 as u32,
                588 as u32,
            ],
            (2, 3),
            &get_device(),
        )
        .unwrap();
        let targets = Tensor::from_vec(
            vec![
                3626 as u32,
                6100 as u32,
                345 as u32,
                1107 as u32,
                588 as u32,
                11311 as u32,
            ],
            (2, 3),
            &get_device(),
        )
        .unwrap();
        let model = DummyGptModel::new(GptConfig {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_layers: 1,
            n_heads: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        })
        .unwrap();
        let logits = model.forward(&inputs).unwrap();
        let probas = softmax(&logits, D::Minus1).unwrap();
        let token_ids = probas.argmax_keepdim(D::Minus1).unwrap();
        let output = token_ids.i((0, .., ..)).unwrap().flatten_all().unwrap();
        let text = token_ids_to_text(&output).unwrap();
        let tar1 = targets
            .i((0, ..))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>();
        let text_idx = 0;
        let target_probas_1 =
            fancy_index(&probas.i(text_idx).unwrap(), &[0, 1, 2], tar1.as_slice()).unwrap();
        let text_idx_1 = 1;
        let tar2 = targets
            .i((0, ..))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>();
        let target_probas_2 =
            fancy_index(&probas.i(text_idx_1).unwrap(), &[0, 1, 2], tar2.as_slice()).unwrap();
        let log_probas = Tensor::log(&Tensor::cat(&[&target_probas_1, &target_probas_2], 0).unwrap()).unwrap();
        println!("log_probas: {:?}", log_probas);
        let cross_entropy = log_probas.mean(D::Minus1).unwrap() * -1.0;
        println!("cross_entropy: {:?}", cross_entropy);
    }


    #[test]
    fn test_cross_entropy() {
        let inputs: Tensor = Tensor::from_vec(
            vec![
                16833 as u32,
                3626 as u32,
                6100 as u32,
                40 as u32,
                1107 as u32,
                588 as u32,
            ],
            (2, 3),
            &get_device(),
        )
        .unwrap();
        let targets = Tensor::from_vec(
            vec![
                3626 as u32,
                6100 as u32,
                345 as u32,
                1107 as u32,
                588 as u32,
                11311 as u32,
            ],
            (2, 3),
            &get_device(),
        )
        .unwrap();
        let model = DummyGptModel::new(GptConfig {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_layers: 1,
            n_heads: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        })
        .unwrap();
        let logits = model.forward(&inputs).unwrap();
        let logits_flat = logits.flatten(0, 1).unwrap();
        let targets_flat = targets.flatten(0, 1).unwrap();
        let loss = cross_entropy(&logits_flat, &targets_flat).unwrap();
        println!("loss: {:?}", loss);
    }


    #[test]
    fn load_the_verdict() {
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        println!("text_data: {:?}", text_data.len());
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let tokens = tokenizer.encode(text_data, false).unwrap();
        println!("tokens: {:?}", tokens.len());
    }

    #[test]
    fn test_dataset() {
        
    }
}

pub fn fancy_index(t: &Tensor, rows: &[usize], cols: &[usize]) -> Result<Tensor> {
    let mut picked = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let val = t.i(rows[i])?.i(cols[i])?;
        picked.push(val.unsqueeze(0)?);
    }

    let result = Tensor::cat(&picked, 0)?;
    Ok(result)
}

