import {nextui} from '@nextui-org/theme';
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
        primary: "#F2D7D5",
        lightSecondary: "#F1B9BB",
        secondary: "#F78F8F",
        accent: "#F25F5C",
        darkAccent: "#CF514F",
      },
      
    },
  },
};
export default config;
