use candle_core::Tensor;

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

        let mut context_vec_2 = Tensor::zeros(query.shape(), DType::F32, &device).unwrap();
        for i in 0..t.shape().dims()[0] {
            let x_i = t.get(i).unwrap();
            println!("{:?}", x_i);
            println!("{:?}", softmax_scores.get(i).unwrap());
            let l = softmax_scores.get(i).unwrap().matmul(&x_i).unwrap() ;
            context_vec_2 = context_vec_2.add(&l).unwrap();
        }
        println!("{:?}", context_vec_2);
    }
}
