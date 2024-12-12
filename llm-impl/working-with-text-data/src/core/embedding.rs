use candle::{DType, Tensor};

struct Embedding {
    pub embedding: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_size: usize) -> Self {
        let embedding = Tensor::randn(
            &[vocab_size, embedding_size],
            DType::F32,
            &Default::default(),
        );
        Self { embedding }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {

        let token_tensor = Tensor::from_vec(input, DType::U32);
    }
}

