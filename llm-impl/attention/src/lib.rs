use candle_core::{Tensor, Device,Result};
use candle_nn::ops::softmax;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn naive_softmax(x: &Tensor) -> Tensor {
    let exp = x.exp().unwrap();
    println!("{:?}", exp);
    let sum = exp.sum_keepdim(0).unwrap();
    println!("{:?}", sum);
    exp.broadcast_div(&sum).unwrap()
}

pub struct SelfAttentionV1 {
    w_query: Tensor,
    w_key: Tensor,
    w_value: Tensor,
    device: Device
}

impl SelfAttentionV1 {
    pub fn new(d_in: usize, d_out: usize) -> Self {
        let device = Device::Cpu;
        let w_query = Tensor::rand(0.0, 1.0, (d_in, d_out), &device).unwrap();
        let w_key = Tensor::rand(0.0, 1.0, (    d_in, d_out), &device).unwrap();
        let w_value = Tensor::rand(0.0, 1.0, (d_in, d_out), &device).unwrap();
        Self {device, w_query, w_key, w_value }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (T, d_in) -> (T, d_out) 

        // Project input into query, key and value vectors
        let queries = self.project_queries(x)?;
        let keys = self.project_keys(x)?;
        let values = self.project_values(x)?;

        // Calculate attention scores and apply scaling
        let attn_scores = self.calculate_attention_scores(&queries, &keys)?;
        let scaled_scores = self.scale_scores(attn_scores)?;

        // Apply softmax to get attention weights and compute final context
        let attn_weights = softmax(&scaled_scores, 1)?;
        let context = attn_weights.matmul(&values)?;

        Ok(context)
    }

    fn project_queries(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.w_query)
    }

    fn project_keys(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.w_key)
    }

    fn project_values(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.w_value)
    }

    fn calculate_attention_scores(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        queries.matmul(&keys.t()?)
    }

    fn scale_scores(&self, scores: Tensor) -> Result<Tensor> {
        let (_, dk) = scores.shape().dims2()?;
        let scale_factor = Tensor::new(vec![dk as f64], &self.device)?.sqrt()?;
        scores.broadcast_div(&scale_factor)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ops::softmax;

    #[test]
    fn test_softmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &Device::Cpu).unwrap();
        let result = naive_softmax(&x);
        println!("{:?}", result);
    }

    #[test]
    fn it_works() {
        let device = Device::Cpu;
        let t = Tensor::new(
            vec![
                vec![0.43, 0.15, 0.89], // Your     (x^1)
                vec![0.55, 0.87, 0.66], // journey  (x^2)
                vec![0.57, 0.85, 0.64], // starts   (x^3)
                vec![0.22, 0.58, 0.33], // with     (x^4)
                vec![0.77, 0.25, 0.10], // one      (x^5)
                vec![0.05, 0.80, 0.55], // step     (x^6)
            ],
            &device,
        )
        .unwrap();

        let query = t.get(1).unwrap();
        let mut attn_scores_2 = Vec::new();
        for i in 0..t.shape().dims()[0] {
            let x_i = t.get(i).unwrap();
            let score = (&x_i * &query).unwrap().sum_all().unwrap();
            attn_scores_2.push(score);
        }
        let attn_scores_2 = Tensor::stack(&attn_scores_2, 0).unwrap();
        let softmax_scores = softmax(&attn_scores_2, 0).unwrap();

        let mut context_vec_2 = Tensor::zeros(query.shape(), DType::F64, &device).unwrap();
        for i in 0..t.shape().dims()[0] {
            let x_i = t.get(i).unwrap();
            println!("{:?}", x_i);
            println!("{:?}", softmax_scores.get(i).unwrap());
            let l = x_i.broadcast_mul(&softmax_scores.get(i).unwrap()).unwrap();
            context_vec_2 = context_vec_2.add(&l).unwrap();
        }
        println!("{:?}", context_vec_2);
    }


    #[test]
    fn it_works_2() {
        let device = Device::Cpu;
        let t = Tensor::new(
            vec![
                vec![0.43, 0.15, 0.89], // Your     (x^1)
                vec![0.55, 0.87, 0.66], // journey  (x^2)
                vec![0.57, 0.85, 0.64], // starts   (x^3)
                vec![0.22, 0.58, 0.33], // with     (x^4)
                vec![0.77, 0.25, 0.10], // one      (x^5)
                vec![0.05, 0.80, 0.55], // step     (x^6)
            ],
            &device,
        )
        .unwrap();
        let mut attn_scores = Vec::new();
        for i in 0..t.shape().dims()[0] {
            let mut row_scores = Vec::new();
            for j in 0..t.shape().dims()[0] {
                let x_i = t.get(i).unwrap();
                let x_j = t.get(j).unwrap();
                let score = (&x_i * &x_j).unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap();
                row_scores.push(score);
            }
            attn_scores.push(row_scores);
        }
        let attn_scores = Tensor::new(attn_scores, &device).unwrap();
        println!("{}", attn_scores);
    }

    #[test]
    fn it_works_3() {
        let device = Device::Cpu;
        let t = Tensor::new(
            vec![
                vec![0.43, 0.15, 0.89], // Your     (x^1)
                vec![0.55, 0.87, 0.66], // journey  (x^2)
                vec![0.57, 0.85, 0.64], // starts   (x^3)
                vec![0.22, 0.58, 0.33], // with     (x^4)
                vec![0.77, 0.25, 0.10], // one      (x^5)
                vec![0.05, 0.80, 0.55], // step     (x^6)
            ],
            &device,
        )
        .unwrap();
        let attn_scores = t.matmul(&t.t().unwrap()).unwrap();
        let softmax_scores = softmax(&attn_scores, 1).unwrap();
        let all_context_vecs =  softmax_scores.matmul(&t).unwrap();
        println!("{}", all_context_vecs)
    }

    #[test]
    fn test_self_attention_v1() {
        let device = Device::Cpu;
        let t = Tensor::new(
            vec![
                vec![0.43, 0.15, 0.89], // Your     (x^1)
                vec![0.55, 0.87, 0.66], // journey  (x^2)
                vec![0.57, 0.85, 0.64], // starts   (x^3)
                vec![0.22, 0.58, 0.33], // with     (x^4)
                vec![0.77, 0.25, 0.10], // one      (x^5)
                vec![0.05, 0.80, 0.55], // step     (x^6)
            ],
            &device,
        )
        .unwrap();
        let d_in = t.shape().dims()[1];
        let d_out = 2;
        // (6, 3) -> (6, 2)
        let self_attn = SelfAttentionV1::new(d_in, d_out);
        let out = self_attn.forward(&t).unwrap();
        println!("{}", out);

    }
}
