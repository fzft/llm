use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_datasets::{batcher::Iter2, Batcher};
use candle_nn::loss::{self, cross_entropy};
use candle_nn::optim::Optimizer;
use gpt::{get_device, DummyGptModel};
use tokenizers::Tokenizer;
use working_with_text_data::Dataset;

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

pub fn split_text(text: &str, split_ratio: f32) -> (String, String) {
    let split_point = (text.len() as f32 * split_ratio) as usize;
    let first = text[..split_point].to_string();
    let second = text[split_point..].to_string();
    (first, second)
}

pub fn clac_loss_batch(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &DummyGptModel,
    device: &Device,
) -> Result<Tensor> {
    let input_batch = input_batch.to_device(device).unwrap();
    let target_batch = target_batch.to_device(device).unwrap();
    let logits = model.forward(&input_batch).unwrap();
    let loss = cross_entropy(
        &logits.flatten(0, 1).unwrap(),
        &target_batch.flatten_all().unwrap(),
    )
    .unwrap();
    Ok(loss)
}

pub fn clac_loss_loader(
    loader: Batcher<Iter2<Dataset>>,
    model: &DummyGptModel,
    device: &Device,
    num_batches: usize,
) -> Result<f32> {
    let mut total_loss = Tensor::zeros((), DType::F32, device).unwrap();
    for batch in loader {
        let (input_batch, target_batch) = batch.unwrap();
        let loss = clac_loss_batch(&input_batch, &target_batch, model, device)?;
        total_loss = (&total_loss + &loss)?;
    }
    Ok(total_loss.to_scalar::<f32>().unwrap() / num_batches as f32)
}

pub fn train_model_sample<T: Optimizer>(
    model: &DummyGptModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: &mut T,
    device: &Device,
    num_epochs: usize,
) -> Result<()> {
    let train_losses = Vec::<f32>::new();
    let val_losses = Vec::<f32>::new();
    for epoch in 0..num_epochs {
        let train_loader = Batcher::new2(train_dataset.clone()).batch_size(2);
        let val_loader = Batcher::new2(val_dataset.clone()).batch_size(2);
        for batch in train_loader {
            let (input_batch, target_batch) = batch.unwrap();
            let loss = clac_loss_batch(&input_batch, &target_batch, model, device)?;
            let grads = loss.backward()?;
            optimizer.step(&grads)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use candle_datasets::Batcher;
    use candle_nn::loss::cross_entropy;
    use candle_nn::ops::softmax;
    use candle_nn::{VarBuilder, VarMap};
    use gpt::*;
    use working_with_text_data::Dataset;

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
    fn test_split_text() {
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        let (first, second) = split_text(&text_data, 0.5);
        println!("first: {:?}", first.len());
        println!("second: {:?}", second.len());
    }

    #[test]
    fn test_generate_text() {
        let device = get_device();
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
            &device,
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
            &device,
        )
        .unwrap();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = DummyGptModel::new(
            vs,
            GptConfig {
                vocab_size: 50257,
                context_length: 1024,
                emb_dim: 768,
                n_layers: 1,
                n_heads: 12,
                drop_rate: 0.1,
                qkv_bias: false,
            },
        )
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
        let log_probas =
            Tensor::log(&Tensor::cat(&[&target_probas_1, &target_probas_2], 0).unwrap()).unwrap();
        println!("log_probas: {:?}", log_probas);
        let cross_entropy = log_probas.mean(D::Minus1).unwrap() * -1.0;
        println!("cross_entropy: {:?}", cross_entropy);
    }

    #[test]
    fn test_cross_entropy() {
        let device = get_device();
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
            &device,
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
            &device,
        )
        .unwrap();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = DummyGptModel::new(
            vs,
            GptConfig {
                vocab_size: 50257,
                context_length: 1024,
                emb_dim: 768,
                n_layers: 1,
                n_heads: 12,
                drop_rate: 0.1,
                qkv_bias: false,
            },
        )
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
        let device = get_device();
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let (first, second) = split_text(&text_data, 0.9);
        let train_dataset = Dataset::new(&first, &tokenizer, 256, 256);
        let test_dataset = Dataset::new(&second, &tokenizer, 256, 256);
        let train_dataloader = Batcher::new2(train_dataset).batch_size(2);
        let test_dataloader = Batcher::new2(test_dataset).batch_size(2);
        println!("train_dataloader");
        for batch in train_dataloader {
            let (inputs, targets) = batch.unwrap();
            let inputs = inputs.to_device(&device).unwrap();
            let targets = targets.to_device(&device).unwrap();
            println!(
                "inputs shape: {:?}, targets shape: {:?}",
                inputs.shape(),
                targets.shape()
            );
        }
        println!("test_dataloader");
        for batch in test_dataloader {
            let (inputs, targets) = batch.unwrap();
            let inputs = inputs.to_device(&device).unwrap();
            let targets = targets.to_device(&device).unwrap();
            println!(
                "inputs shape: {:?}, targets shape: {:?}",
                inputs.shape(),
                targets.shape()
            );
        }
    }

    #[test]
    fn test_clac_loss_loader() {
        let device = get_device();
        let file_path = "../DATA/the-verdict.txt";
        let text_data = std::fs::read_to_string(file_path).unwrap();
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let (first, second) = split_text(&text_data, 0.9);
        let train_dataset = Dataset::new(&first, &tokenizer, 64, 64);
        let test_dataset = Dataset::new(&second, &tokenizer, 64, 64);
        let train_dataloader = Batcher::new2(train_dataset).batch_size(2);
        let test_dataloader = Batcher::new2(test_dataset).batch_size(2);
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = DummyGptModel::new(
            vs,
            GptConfig {
                vocab_size: 50257,
                context_length: 64,
                emb_dim: 768,
                n_layers: 12,
                n_heads: 12,
                drop_rate: 0.1,
                qkv_bias: false,
            },
        )
        .unwrap();
        let loss = clac_loss_loader(train_dataloader, &model, &device, 9).unwrap();
        println!("loss: {:?}", loss);
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
