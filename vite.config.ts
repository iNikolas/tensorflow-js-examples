import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import circleDependency from "vite-plugin-circular-dependency";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [tsconfigPaths(), react(), tailwindcss(), circleDependency()],
    base: `/${env.VITE_BASE_PATH}`,
  };
});
