import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite' // 這是我們剛剛安裝的新橋樑

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(), // 在這裡呼叫它，讓 Vite 認識 Tailwind
  ],
})
