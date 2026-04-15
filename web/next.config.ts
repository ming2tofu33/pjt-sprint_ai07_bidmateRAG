import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Next.js detects the parent repo's package.json (kordoc CLI) as a workspace
  // root and prints a warning. Pin tracing root to this web/ directory.
  outputFileTracingRoot: path.join(__dirname),
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8100/api/:path*",
      },
    ];
  },
};

export default nextConfig;
