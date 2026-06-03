import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useProxy = env.VITE_RELEASEFRAME_USE_PROXY === "1";
  const proxyTarget =
    env.VITE_PROXY_API_TARGET || "http://localhost:8000";

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src"),
      },
    },
    server: {
      port: 5173,
      proxy: useProxy
        ? {
            "/api": {
              target: proxyTarget,
              changeOrigin: true,
              secure: false,
              rewrite: (p) => p.replace(/^\/api/, ""),
              headers: {
                "ngrok-skip-browser-warning": "1",
              },
            },
          }
        : undefined,
    },
  };
});
