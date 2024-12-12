use std::thread;


fn main() {
    let data = vec![1, 2, 3];
    let mut sum = 0;

    // Start a scoped thread that allows us to use `data` and `sum` across multiple threads
    thread::scope(|s| {
        for &item in &data {
            s.spawn(move || {
                sum += item;
            });
        }
    });

    // All threads finish executing here
    println!("Total sum: {}", sum); // Should print "Total sum: 6"
}
