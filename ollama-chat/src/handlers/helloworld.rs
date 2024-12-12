use tonic::{Request, Response, Status};

pub mod helloworld {
    tonic::include_proto!("helloworld");
}

pub use helloworld::{
    greeter_server::{Greeter, GreeterServer},
    HelloRequest, HelloReply
};

#[derive(Debug, Default)]
pub struct MyGreeter {}

#[tonic::async_trait]
impl Greeter for MyGreeter {
    async fn say_hello(&self, request: Request<HelloRequest>) -> Result<Response<HelloReply>, Status> {
        let reply = HelloReply { message: format!("Hello, {}!", request.into_inner().name) };
        Ok(Response::new(reply))
    }
}
