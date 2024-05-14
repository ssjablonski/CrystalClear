import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // primary: "#222629",
        // lightSecondary: "#6b6e70",
        // secondary: "#474b4f",
        // accent: "#86c232",
        // darkAccent: "#61892f",

        // primary: "#ccdbdc",
        // lightSecondary: "#9ad1d4",
        // secondary: "#80ced7",
        // accent: "#007ea7",
        // darkAccent: "#003249",

        primary: "#F2D7D5",
        lightSecondary: "#F1B9BB",
        secondary: "#F78F8F",
        accent: "#F25F5C",
        darkAccent: "#CF514F",


      },
      
    },
  },
  plugins: [],
};
export default config;
