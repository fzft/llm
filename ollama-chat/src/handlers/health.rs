
#[utoipa::path(get, path = "/health", responses(
    (status = 200, description = "OK"),
    (status = 500, description = "Internal Server Error"),
), tag="Health")]
pub async fn health() -> &'static str {
    "OK"
}
