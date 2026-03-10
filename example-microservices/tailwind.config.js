/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'kku-gold': '#ffbb1c',      // สีทอง มข.
        'kku-maroon': '#791d2b',    // สีแดงเลือดหมู มข.
      },
    },
  },
  plugins: [],
}