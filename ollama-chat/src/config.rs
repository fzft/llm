use std::fs::File;
use std::sync::OnceLock;
use serde::Deserialize;

static GLOBAL_CONFIG_VAR: OnceLock<Config> = OnceLock::new();

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct Config {
    pub app_name: String,
    pub debug_mode: bool,
    pub server: ServerConfig,
    pub ollama: OllamaConfig,
    pub log: LogConfig,
}

#[derive(Deserialize, Debug)]
pub struct ServerConfig {
    pub http_host: String,
    pub http_port: u16,
    pub grpc_host: String,
    pub grpc_port: u16,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
    pub model: String,
}

#[derive(Deserialize, Debug)]
pub struct LogConfig {
    pub file: String,
}

pub fn load_config() -> &'static Config {
    GLOBAL_CONFIG_VAR.get_or_init(|| {
        let config_path = "config.yaml";
        let file = File::open(config_path).expect("Failed to open config file");
        serde_yaml::from_reader(file).expect("Failed to parse config file")
    })
}
