build:
    docker buildx build \
        --builder multiarch-builder \
        --platform linux/arm64 \
        --load \
        -t ghcr.io/dav354/rps:latest \
        game_server

build_and_publish:
    just build
    docker push ghcr.io/dav354/rps:latest

build_and_pi:
    just build
    docker save ghcr.io/dav354/rps:latest | ssh david@10.77.77.3 "docker load"