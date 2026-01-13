import { useEffect, useState } from 'react'
import { getVersionInfo } from '../lib/api'

export default function Footer() {
  const [versionInfo, setVersionInfo] = useState('Checking system...')

  useEffect(() => {
    getVersionInfo()
      .then((data) => {
        setVersionInfo(`PyTorch ${data.pytorch_version} • CUDA ${data.cuda_version}`)
      })
      .catch(() => {
        setVersionInfo('System Ready')
      })
  }, [])

  return (
    <footer className="border-t border-white/5 bg-iris-panel/50 mt-auto">
      <div className="container mx-auto px-6 py-6 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="text-xs text-zinc-600">
          &copy; 2024 I.R.I.S. Project. Local AI Rendering.
        </div>
        <div className="text-xs text-zinc-600 font-mono bg-white/5 px-3 py-1.5 rounded-lg border border-white/5">
          {versionInfo}
        </div>
      </div>
    </footer>
  )
}
