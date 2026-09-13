# IntelliGround React Frontend

The IntelliGround frontend is a React + TypeScript + Vite application that provides the user interface for the Evidence-Based RAG system.

It communicates with the FastAPI backend through authenticated HTTP APIs and is served in production using Nginx inside Docker.

## Architecture

Browser
  |
  v
React + TypeScript + Vite
  |
  | HTTP API
  v
FastAPI Backend
  |
  +-- Authentication
  +-- Document Upload
  +-- Document Processing
  +-- Hybrid RAG
  +-- Evidence
  +-- Security / Guardrails
  +-- LLM Evaluation
  +-- Human Review
  +-- Developer Experience

## Run locally

Use Node.js 22.12+ or a compatible newer version.

    npm ci
    npm run dev

Open http://localhost:5173.

## Environment

The browser-facing backend URL is configured through VITE_API_BASE_URL.

Example:

    VITE_API_BASE_URL=http://localhost:8000/api

Never put passwords, authentication tokens, OpenAI API keys, or other secrets in a VITE_* variable because Vite exposes these values to the browser.

## Verification

    npm run lint
    npm run build
    npx playwright install chromium
    npm run test:e2e

## Docker

The production frontend is built using the root project's Dockerfile.frontend.

The Docker image:

1. Installs React dependencies.
2. Builds the production application.
3. Copies the generated dist/ files into Nginx.
4. Serves the application on port 8501.

The Docker Compose service is rag_frontend.

## Application flow

Landing
  |
  v
Login / Signup
  |
  v
Access Setup
  |
  v
Upload Documents
  |
  v
Processing
  |
  v
Main Workspace
  +-- RAG
  +-- Security Lab
  +-- LLM Evaluation
  +-- Human Review
  +-- Developer Experience

Authentication and authorization remain enforced by the FastAPI backend.

RAG credentials and API keys used by the application are not persisted in browser localStorage or sessionStorage.

## Structure

frontend/react-app/
  src/
    components/
    lib/
    pages/
    App.tsx
    main.tsx
    styles/
  e2e/
  nginx.conf
  package.json
  package-lock.json
  vite.config.ts
  playwright.config.ts
