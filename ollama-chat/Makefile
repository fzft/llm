run:
	RUST_BACKTRACE=1 cargo run 
	
docker-build:
	docker build -t ollama-chat:latest .

docker-run:
	docker run -p 3000:3000 ollama-chat:latest

test:
	cargo test

lint:
	cargo clippy --fix --allow-dirty

format:
	cargo fmt
