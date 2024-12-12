use helloworld::greeter_client::GreeterClient;
use helloworld::HelloRequest;

pub mod helloworld {
    tonic::include_proto!("helloworld"); // The generated gRPC client/server code from proto
}

pub async fn say_hello(client: &mut GreeterClient<tonic::transport::Channel>, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let request = tonic::Request::new(HelloRequest {
        name: name.to_string(),
    });

    let response = client.say_hello(request).await?;

    Ok(response.into_inner().message)
}

#[cfg(test)]
mod grpc_helloworld_test {
    use super::*;

    #[tokio::test]
    async fn test_greeter_client_with_real_server() -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("grpc://{}:{}", "0.0.0.0", "50051");
        
        // Create a real client connected to the in-process server
        let mut client = GreeterClient::connect(addr).await?;

        // Call the client function to test
        let response = say_hello(&mut client, "TestUser").await?;

        println!("Response: {}", response);

        assert_eq!(response, "Hello, TestUser!");

        Ok(())
    }
}

