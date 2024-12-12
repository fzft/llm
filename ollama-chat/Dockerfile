FROM rust:1.81 as builder

WORKDIR /app

COPY . .

# Generate the gRPC code, install the protoc compiler
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN cargo build --release

FROM rust:1.81 as runner

WORKDIR /app

COPY --from=builder /app/target/release/ollama-chat .

COPY --from=builder /app/config.yaml .

EXPOSE 3000

CMD ["./ollama-chat"]