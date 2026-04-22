/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Outfit', 'sans-serif'],
      },
      colors: {
        dark: {
          900: '#0B0F19', // Deep dark blue/black background
          800: '#141B2D', // Slightly lighter for panels
          700: '#1E293B',
        },
        primary: {
          500: '#0ea5e9', // Cyan
          400: '#38bdf8',
        },
        accent: {
          500: '#10B981', // Emerald
          400: '#34D399',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
