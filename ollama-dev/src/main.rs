fn returns_a_closure(input: &str) -> impl Fn(i32) -> i32 {
    match input {
        "double" => |mut number| {
            number *= 2;
            println!("Doubling number. Now it is {number}");
            number
        },
        "triple" => |mut number| {
            number *= 3;
            println!("Tripling number. Now it is {number}");
            number
        },
        _ => |number| {
            println!("Sorry, it's the same: {number}.");
            number
        },
    }
}
 
fn main() {
    let my_number = 10;
 
    let mut doubles = returns_a_closure("double");
    let mut triples = returns_a_closure("triple");
    let mut does_nothing = returns_a_closure("HI");
 
    let doubled = doubles(my_number);
    let tripled = triples(my_number);
    let same = does_nothing(my_number);
}


