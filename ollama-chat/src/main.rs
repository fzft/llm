mod config;
mod handlers;

use std::fs::OpenOptions;
use std::io::Write;
use axum::{
    extract::Request,
    middleware::{self, Next},
    response::Response,
    routing::any,
};

use tonic::transport::Server;
use std::net::{SocketAddr, Ipv4Addr};
use tokio::signal;
use log::*;
use env_logger::Builder;
use chrono::Local;
use utoipa::OpenApi;
use utoipa_axum::router::{self, OpenApiRouter};
use utoipa_axum::routes;
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    tags(
        (name = "Health", description = "Health check API endpoints"),
        (name = "System Info", description = "System Info API endpoints"),
        (name = "Chat", description = "Chat API endpoints")
    )
)]
struct ApiDoc;

#[tokio::main]
async fn main() {
    let config = config::load_config();
    // Open (or create) a file to write logs.
    let log_file = OpenOptions::new()
    .append(true) // Append to the file if it exists
    .create(true) // Create the file if it doesn't exist
    .open(config.log.file.clone())
    .unwrap();

    let target = Box::new(log_file);

    Builder::new()
        .format(move |buf: &mut env_logger::fmt::Formatter, record| {
            let log_entry = format!(
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"), // Local timestamp
                record.level(),
                record.args()
            );

            println!("{}", log_entry);
            writeln!(
               buf,
               "{}",
               log_entry
            )
        })
        .target(env_logger::Target::Pipe(target))
        .filter(None, LevelFilter::Info)
        .init();

    let grpc_service = handlers::helloworld::GreeterServer::new(handlers::helloworld::MyGreeter::default());

    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(handlers::health))
        .nest("/api/metrics", handlers::metrics::router())
        .nest("/api/llm", handlers::chat::router())
        .layer(middleware::from_fn(print_request_response))
        .split_for_parts();

    let router = router.route("/stream", any(handlers::chat::streaming_completion));
    let router = router.merge(SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", api));
    
    let addr = SocketAddr::from((config.server.http_host.parse::<Ipv4Addr>().unwrap(), config.server.http_port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    
    // Start the gRPC server
    tokio::spawn(async move {
       info!("Starting gRPC server on {}:{}", config.server.grpc_host, config.server.grpc_port);
       Server::builder()
            .add_service(grpc_service)
            .serve(format!("{}:{}", config.server.grpc_host, config.server.grpc_port).parse().unwrap())
            .await
            .unwrap();
    });
    
    info!("Starting server on {}:{}", config.server.http_host, config.server.http_port);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await.unwrap();
}

async fn print_request_response(
    req: Request,
    next: Next,
) -> Response {
    info!("Request: {} {}", req.uri(), req.method());
    let res = next.run(req).await;
    info!("Response: {}", res.status());
    res
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}