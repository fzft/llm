use std::ops::ControlFlow;

use log::info;
use serde::Deserialize;
use serde::Serialize;
use crate::config;
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade, Message},
    response::IntoResponse,
    http::StatusCode,
    Json
};
use futures::{sink::SinkExt, stream::StreamExt};
use tokio::sync::mpsc;


#[derive(Deserialize, Debug, utoipa::ToSchema)]
pub struct CompletionRequest {
    pub prompt: String,
}

#[derive(Serialize, Debug, utoipa::ToSchema)]
pub struct CompletionResponse {
    pub response: String,
}

pub fn router() -> OpenApiRouter {
    OpenApiRouter::new().routes(routes!(completion))
}

#[utoipa::path(
    post,
    path = "/generate",
    request_body = CompletionRequest,
    responses( (status = 200, body = CompletionResponse)),
    tag = "Chat"
)]
pub async fn completion(body: Json<CompletionRequest>) -> impl IntoResponse {
    let config: &config::Config = config::load_config();
    info!("ollama host: {}, port: {}, model: {}", config.ollama.host, config.ollama.port, config.ollama.model);
    let ollama = Ollama::new(config.ollama.host.to_string(), config.ollama.port);
    let response = ollama.generate(GenerationRequest::new(config.ollama.model.to_string(), body.prompt.to_string())).await;
    if let Ok(response) = response {
        (StatusCode::OK, Json(CompletionResponse {
            response: response.response,
        }))
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(CompletionResponse {
            response: "Failed to generate completion".to_string(),
        }))
    }
}

pub async fn streaming_completion(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(stream_response)
}

async fn stream_response(socket: WebSocket) {
    info!("WebSocket connection established");
    let (mut tx, mut rx) = mpsc::unbounded_channel();
    let (mut sender, mut receiver) = socket.split();
    let config: &config::Config = config::load_config();
    let ollama = Ollama::new(config.ollama.host.to_string(), config.ollama.port);
     // Spawn a task to send messages from the channel to the websocket
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

   let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if process_message(msg, &mut tx, &ollama, config).await.is_break() {
                break;
            }
        }
   });

   // Wait for the tasks to complete
   tokio::select! {
        _ = (&mut send_task) => {
            info!("send_task finished");
            recv_task.abort();
        },
        _ = (&mut recv_task) => {
            info!("recv_task finished");
            send_task.abort();
        },
   }

   info!("WebSocket connection closed");
}


async fn process_message(message: Message, tx: &mut mpsc::UnboundedSender<String>, ollama: &Ollama, config: &config::Config) -> ControlFlow<(),()> {
    match message {
        Message::Text(text) => {
            info!("Received message: {}", text);
            let mut stream = ollama.generate_stream(GenerationRequest::new(config.ollama.model.to_string(), text)).await.unwrap();
            while let Some(res) = stream.next().await {
                let responses = res.unwrap();
                for resp in responses {
                     match tx.send(resp.response.as_str().to_string()) {
                         Ok(_) => (),
                         Err(e) => {
                             eprintln!("Failed to send message: {}", e);
                             break;
                         }
                     }
                }
            }
            ControlFlow::Continue(())
        },
        _ => ControlFlow::Continue(()),
    }
}