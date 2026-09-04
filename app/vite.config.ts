import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: { exclude: ["maplibre-gl"] },
  server: {
    proxy: {
      "/events": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/replay": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/metrics": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/predict": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/timeline": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/verify-image": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
