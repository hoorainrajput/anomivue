import type { Config } from "tailwindcss";

/**
 * ANOMIVUE colour theme
 * Pale Silver      #CDC8C2
 * Cadet Blue       #A2B2C1
 * Indigo Dye       #304961
 * Black Coral      #3C5369
 * Rich Black       #090E12
 *
 * Purane code mein slate / indigo / purple / sky ki classes use hui hain.
 * Isliye yahan unhi ke shades is palette se replace kar diye hain,
 * taake poora UI sirf is ek file se badal jaye.
 */

const slate = {
  50: "#F4F2F0",
  100: "#E6E3DF",
  200: "#CDC8C2", // Pale Silver
  300: "#B3BCC4",
  400: "#8C9DAE",
  500: "#7A8DA1",
  600: "#3C5369", // Black Coral
  700: "#2C4156",
  800: "#1D2D3D",
  900: "#111B25",
  950: "#090E12", // Rich Black
};

const indigo = {
  50: "#EDF2F6",
  100: "#D9E3EC",
  200: "#C0D0DE",
  300: "#B9C9D8",
  400: "#A2B2C1", // Cadet Blue
  500: "#4A6580",
  600: "#3C5369", // Black Coral
  700: "#304961", // Indigo Dye
  800: "#233749",
  900: "#182736",
  950: "#0F1A24",
};

const purple = {
  50: "#F7F6F4",
  100: "#EEECE9",
  200: "#E3E0DB",
  300: "#DAD6D1",
  400: "#CDC8C2", // Pale Silver (warm accent)
  500: "#A9A49E",
  600: "#857F79",
  700: "#625E5A",
  800: "#42403D",
  900: "#2B2927",
  950: "#191817",
};

const sky = {
  50: "#F0F4F8",
  100: "#E1E9F0",
  200: "#CEDAE5",
  300: "#C2D0DD",
  400: "#B4C6D6",
  500: "#91A7BC",
  600: "#6F889F",
  700: "#546C82",
  800: "#3C5369",
  900: "#2A3C4E",
  950: "#1A2631",
};

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        slate,
        indigo,
        purple,
        sky,
        background: "#090E12",
        surface: "#0F1820",
        "surface-card": "#14202B",
        "surface-border": "#1D2D3D",
        primary: {
          50: indigo[50],
          100: indigo[100],
          400: indigo[400],
          500: indigo[500],
          600: indigo[600],
          700: indigo[700],
        },
        accent: {
          blue: "#A2B2C1",
          cyan: "#8C9DAE",
          purple: "#CDC8C2",
          pink: "#B8A9A3",
          // Alert colours jaan boojh kar nahi badle: anomaly ko laal hi dikhna chahiye
          emerald: "#10b981",
          amber: "#f59e0b",
          rose: "#f43f5e",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 20px -5px rgba(162, 178, 193, 0.28)",
        "glow-rose": "0 0 20px -5px rgba(244, 63, 94, 0.3)",
        card: "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
      },
    },
  },
  plugins: [],
};
export default config;