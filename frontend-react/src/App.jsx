import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import GeneratePage from './pages/GeneratePage'
import GalleryPage from './pages/GalleryPage'
import SettingsPage from './pages/SettingsPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/generate" element={<GeneratePage />} />
      <Route path="/gallery" element={<GalleryPage />} />
      <Route path="/settings" element={<SettingsPage />} />
    </Routes>
  )
}

export default App
