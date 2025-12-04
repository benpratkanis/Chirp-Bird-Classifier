#!/bin/bash

# Deployment Script for Low-Memory Environments (e.g., AWS t2.micro)
# Builds containers sequentially to prevent freezing.

echo "🚀 Starting Deployment..."

# 1. Build Backend
echo "📦 Building Backend..."
docker compose -f docker-compose.prod.yml build backend

# 2. Build Frontend
echo "📦 Building Frontend..."
docker compose -f docker-compose.prod.yml build frontend

# 3. Start Services
echo "🔥 Starting Services..."
docker compose -f docker-compose.prod.yml up -d

echo "5. Cleaning up old images..."
docker image prune -f

echo "✅ Deployment Complete!"
echo "Frontend running on port 3000 (Internal Nginx)"
echo "Backend running on port 8000"
