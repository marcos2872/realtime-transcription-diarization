# Caddyfile is baked into the image: bind-mounting a single file breaks
# under rootless docker user namespaces (permission denied on read).
FROM caddy:2-alpine
COPY Caddyfile /etc/caddy/Caddyfile
