import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    resolveAlias: {
      tailwindcss: new URL("node_modules/tailwindcss", import.meta.url).pathname,
    },
  },
  allowedDevOrigins: ["192.168.5.70"],
};

export default nextConfig;
