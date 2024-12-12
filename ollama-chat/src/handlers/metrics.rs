use sysinfo::System;
use serde_json::Value;
use axum::Json;
use std::collections::HashMap;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

pub fn router() -> OpenApiRouter {
    OpenApiRouter::new().routes(routes!(system_info))
}

#[utoipa::path(get, path = "/system-info", responses(
    (status = 200, description = "OK", body = HashMap<String, Value>),
    (status = 500, description = "Internal Server Error"),
), tag="System Info")]
pub async fn system_info() -> Json<HashMap<String, Value>> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let total_memory = sys.total_memory();
    let used_memory = sys.used_memory();

    sys.refresh_cpu_usage();

    Json(HashMap::from([
        ("total_memory".to_string(), Value::Number(total_memory.into())),
        ("used_memory".to_string(), Value::Number(used_memory.into())),
    ]))
}
