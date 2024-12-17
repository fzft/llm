pub mod gpt;

use gpt::*;

const GPT_CONFIG_124M: GptConfig = GptConfig {
    vocab_size: 50257,    // Vocabulary size
    context_length: 1024, // Context length
    emb_dim: 768,         // Embedding dimension
    n_heads: 12,          // Number of attention heads
    n_layers: 12,         // Number of layers
    drop_rate: 0.1,       // Dropout rate
    qkv_bias: false,      // Query-Key-Value bias
};
