# Front de teste (Next.js standalone). A URL da API chega via build-arg e
# pode ser sobrescrita em runtime? Não — NEXT_PUBLIC_* é fixo no build.
# Para apontar para outro host, rebuild com outro NEXT_PUBLIC_API_BASE_URL.
FROM node:22-alpine AS deps
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /web
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
COPY --from=deps /web/node_modules ./node_modules
COPY . .
RUN mkdir -p ./public && npm run build

FROM node:22-alpine AS runner
WORKDIR /web
ENV NODE_ENV=production
RUN addgroup -S app && adduser -S app -G app
COPY --from=builder --chown=app:app /web/.next/standalone ./
COPY --from=builder --chown=app:app /web/.next/static ./.next/static
COPY --from=builder --chown=app:app /web/public ./public
USER app
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["node", "server.js"]
