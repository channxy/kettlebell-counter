import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Athletic/Sport-inspired dark theme
        kb: {
          bg: "#0a0a0f",
          surface: "#12121a",
          card: "#1a1a24",
          border: "#2a2a3a",
          accent: "#ff6b35",
          "accent-glow": "#ff6b3520",
          success: "#10b981",
          danger: "#ef4444",
          warning: "#f59e0b",
          muted: "#6b7280",
        },
      },
      fontFamily: {
        display: ["var(--font-archivo)", "system-ui", "sans-serif"],
        body: ["var(--font-dm-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "monospace"],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "noise": "url('/noise.svg')",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "count-up": "countUp 0.5s ease-out forwards",
        "slide-up": "slideUp 0.4s ease-out forwards",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        countUp: {
          "0%": { transform: "translateY(100%)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(20px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        glow: {
          "0%": { boxShadow: "0 0 20px rgba(255, 107, 53, 0.3)" },
          "100%": { boxShadow: "0 0 40px rgba(255, 107, 53, 0.6)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;

