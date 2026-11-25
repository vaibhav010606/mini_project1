import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React dev server (port 5173) forwards /api and /files to the FastAPI backend (port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/files": "http://127.0.0.1:8000",
    },
  },
});
