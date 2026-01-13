import { useEffect, useState, useRef } from 'react'
import Sidebar from '../components/Sidebar'
import { getGpuInfo, getVersionInfo, getOutputGallery, getSettings, saveSettings } from '../lib/api'

const API_BASE = ''

export default function SettingsPage() {
  const [gpuInfo, setGpuInfo] = useState(null)
  const [versionInfo, setVersionInfo] = useState(null)
  const [stats, setStats] = useState({ images: 0, storage: '0 MB' })
  const [dramEnabled, setDramEnabled] = useState(false)
  const [vramThreshold, setVramThreshold] = useState(6)
  const [maxDram, setMaxDram] = useState(16)
  const [discordEnabled, setDiscordEnabled] = useState(false)
  const [discordStatus, setDiscordStatus] = useState('Not configured')
  const [nsfwFilterEnabled, setNsfwFilterEnabled] = useState(true)
  const [nsfwStrength, setNsfwStrength] = useState(2)
  const [saving, setSaving] = useState(false)
  const lastSettingsRef = useRef(null)
  
  // Device switching
  const [currentDevice, setCurrentDevice] = useState('cuda')
  const [devices, setDevices] = useState([])
  const [deviceSwitching, setDeviceSwitching] = useState(false)
  const [deviceStatus, setDeviceStatus] = useState('')

  useEffect(() => {
    loadData()
    loadSettingsFromServer()
    loadDiscordStatus()
    loadDevices()
    const gpuInterval = setInterval(loadGpuInfo, 3000)
    // Poll settings every 2 seconds for live sync with HTML frontend
    const settingsInterval = setInterval(loadSettingsFromServer, 2000)
    // Poll Discord status every 5 seconds
    const discordInterval = setInterval(loadDiscordStatus, 5000)
    return () => {
      clearInterval(gpuInterval)
      clearInterval(settingsInterval)
      clearInterval(discordInterval)
    }
  }, [])

  async function loadSettingsFromServer() {
    try {
      const data = await getSettings()
      const settings = data.settings || data
      // Only update if settings changed (to avoid UI flicker)
      const settingsStr = JSON.stringify(settings)
      if (lastSettingsRef.current === settingsStr) return
      lastSettingsRef.current = settingsStr
      
      if (settings.nsfwEnabled !== undefined) setNsfwFilterEnabled(settings.nsfwEnabled)
      if (settings.nsfwStrength !== undefined) setNsfwStrength(settings.nsfwStrength)
      if (settings.dramEnabled !== undefined) setDramEnabled(settings.dramEnabled)
      if (settings.vramThreshold !== undefined) setVramThreshold(settings.vramThreshold)
      if (settings.maxDram !== undefined) setMaxDram(settings.maxDram)
      if (settings.discordEnabled !== undefined) setDiscordEnabled(settings.discordEnabled)
    } catch (e) { console.error('Failed to load settings:', e) }
  }

  async function handleSaveSettings() {
    setSaving(true)
    try {
      await saveSettings({
        nsfwEnabled: nsfwFilterEnabled,
        nsfwStrength: nsfwStrength,
        dramEnabled: dramEnabled,
        vramThreshold: vramThreshold,
        maxDram: maxDram,
        discordEnabled: discordEnabled,
      })
      // Update ref so polling doesn't overwrite immediately
      lastSettingsRef.current = JSON.stringify({
        nsfwEnabled: nsfwFilterEnabled,
        nsfwStrength: nsfwStrength,
        dramEnabled: dramEnabled,
        vramThreshold: vramThreshold,
        maxDram: maxDram,
        discordEnabled: discordEnabled,
      })
    } catch (e) { console.error('Failed to save settings:', e) }
    setSaving(false)
  }

  async function handleDiscordToggle(enabled) {
    setDiscordEnabled(enabled)
    if (enabled) {
      setDiscordStatus('Starting...')
      try {
        const response = await fetch('/api/discord-bot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: true })
        })
        const data = await response.json()
        if (data.success) {
          setDiscordStatus('Running')
        } else {
          // Show specific error message
          const errorMsg = data.message || 'Failed'
          setDiscordStatus(errorMsg === 'Bot token not configured' ? 'Token missing' : errorMsg)
          setDiscordEnabled(false)
        }
      } catch (e) {
        setDiscordStatus('Server offline')
        setDiscordEnabled(false)
      }
    } else {
      setDiscordStatus('Stopping...')
      try {
        await fetch('/api/discord-bot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: false })
        })
        setDiscordStatus('Not configured')
      } catch (e) {
        setDiscordStatus('Server offline')
      }
    }
  }

  async function loadDiscordStatus() {
    try {
      const response = await fetch('/api/discord-bot/status')
      const data = await response.json()
      if (data.running) {
        setDiscordStatus('Running')
        setDiscordEnabled(true)
      } else if (!discordEnabled) {
        setDiscordStatus('Not configured')
      }
    } catch (e) { /* ignore */ }
  }

  async function loadDevices() {
    try {
      const response = await fetch('/api/devices')
      const data = await response.json()
      setDevices(data.devices || [])
      setCurrentDevice(data.current_device)
    } catch (e) { console.error('Failed to load devices:', e) }
  }

  async function switchDevice(device) {
    if (device === currentDevice || deviceSwitching) return
    
    setDeviceSwitching(true)
    setDeviceStatus('Switching...')
    
    try {
      const response = await fetch(`/api/device?device=${device}`, { method: 'POST' })
      const data = await response.json()
      
      if (data.success) {
        setCurrentDevice(data.new_device)
        setDeviceStatus(data.model_reloaded ? 'Switched & model reloaded' : 'Switched')
        setTimeout(() => setDeviceStatus(''), 3000)
      } else {
        setDeviceStatus('Failed')
      }
    } catch (e) {
      console.error('Device switch error:', e)
      setDeviceStatus('Error')
    }
    
    setDeviceSwitching(false)
    loadDevices()
  }

  async function loadData() {
    loadGpuInfo()
    loadVersionInfo()
    loadStats()
  }

  async function loadGpuInfo() {
    try {
      const data = await getGpuInfo()
      setGpuInfo(data)
    } catch (e) { console.error('GPU info error:', e) }
  }

  async function loadVersionInfo() {
    try {
      const data = await getVersionInfo()
      setVersionInfo(data)
    } catch (e) { console.error('Version info error:', e) }
  }

  async function loadStats() {
    try {
      const data = await getOutputGallery()
      const count = data.images?.length || 0
      setStats({ images: count, storage: `${(count * 1.5).toFixed(1)} MB` })
    } catch (e) { console.error('Stats error:', e) }
  }

  const gpuLoad = gpuInfo?.gpu?.gpu_utilization ?? 0
  const gpuTemp = gpuInfo?.gpu?.gpu_temp ?? 0
  const vramUsed = gpuInfo?.gpu?.vram_used ?? 0
  const vramTotal = gpuInfo?.gpu?.vram_total ?? 4
  const vramPerc = vramTotal > 0 ? (vramUsed / vramTotal * 100) : 0
  const powerDraw = gpuInfo?.gpu?.power_draw ?? 0
  const gpuName = gpuInfo?.gpu?.gpu_name ?? '--'
  
  // CPU & RAM
  const cpuPercent = gpuInfo?.gpu?.cpu_percent ?? 0
  const cpuFreq = gpuInfo?.gpu?.cpu_freq ?? 0
  const cpuCores = gpuInfo?.gpu?.cpu_cores ?? 0
  const ramUsed = gpuInfo?.gpu?.ram_used ?? 0
  const ramTotal = gpuInfo?.gpu?.ram_total ?? 0
  const ramPercent = gpuInfo?.gpu?.ram_percent ?? 0

  const getNsfwStrengthLabel = (value) => {
    switch(value) {
      case 1: return 'Relaxed'
      case 2: return 'Standard'
      case 3: return 'Strict'
      default: return 'Standard'
    }
  }

  return (
    <div className="flex h-screen w-full overflow-hidden text-sm">
      <Sidebar>
        {/* Sidebar Content - Quick Actions */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <div className="p-4 space-y-4">
            <div className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wider">Quick Stats</div>
            
            <div className="space-y-3">
              <div className="p-3 bg-iris-card rounded-xl border border-iris-border">
                <div className="text-[10px] text-zinc-500 uppercase mb-1">GPU Load</div>
                <div className="text-xl font-bold text-white font-mono">{Math.round(gpuLoad)}%</div>
                <div className="w-full bg-black/30 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div className="h-full rounded-full transition-all" style={{ width: `${gpuLoad}%`, background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)' }} />
                </div>
              </div>
              
              <div className="p-3 bg-iris-card rounded-xl border border-iris-border">
                <div className="text-[10px] text-zinc-500 uppercase mb-1">VRAM</div>
                <div className="text-xl font-bold text-white font-mono">{vramUsed.toFixed(1)} / {vramTotal.toFixed(1)} GB</div>
                <div className="w-full bg-black/30 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div className="h-full rounded-full transition-all" style={{ width: `${vramPerc}%`, background: 'linear-gradient(90deg, #6366f1, #818cf8)' }} />
                </div>
              </div>

              <div className="p-3 bg-iris-card rounded-xl border border-iris-border">
                <div className="text-[10px] text-zinc-500 uppercase mb-1">Library</div>
                <div className="text-xl font-bold text-white font-mono">{stats.images} images</div>
                <div className="text-xs text-zinc-500 mt-1">{stats.storage} used</div>
              </div>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="p-4 border-t border-iris-border bg-iris-panel">
          <button onClick={handleSaveSettings} disabled={saving} className="btn-primary w-full py-3.5 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2 disabled:opacity-50">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </Sidebar>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-iris-bg">
        <div className="max-w-4xl mx-auto p-8">
          <header className="mb-8">
            <h1 className="text-2xl font-bold text-white mb-2">System Settings</h1>
            <p className="text-zinc-500 text-sm">Manage your hardware resources and system configurations.</p>
          </header>

          <div className="space-y-6">
            {/* Generation Settings */}
            <section className="glass-panel liquid-glass p-6 rounded-2xl">
              <h2 className="text-base font-bold text-white mb-5 flex items-center gap-2">
                <svg className="w-5 h-5 text-iris-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" /></svg>
                Generation Settings
              </h2>

              <div className="space-y-5">
                {/* Compute Device */}
                <div className="p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
                        <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-white">Compute Device</div>
                        <div className="text-xs text-zinc-500">Select GPU or CPU for generation</div>
                      </div>
                    </div>
                    <span className={`text-xs ${deviceStatus.includes('Error') || deviceStatus.includes('Failed') ? 'text-red-400' : deviceStatus ? 'text-yellow-400' : 'text-zinc-500'}`}>
                      {deviceStatus}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {devices.map(device => {
                      const isActive = device.id === currentDevice
                      const colorMap = {
                        nvidia: 'bg-green-600',
                        amd: 'bg-red-600',
                        intel: 'bg-blue-600',
                        apple: 'bg-gray-600',
                        cpu: 'bg-emerald-600'
                      }
                      const labelMap = {
                        nvidia: 'NVIDIA',
                        amd: 'AMD',
                        intel: 'Intel Arc',
                        apple: 'Apple',
                        cpu: 'CPU'
                      }
                      return (
                        <button
                          key={device.id}
                          onClick={() => switchDevice(device.id)}
                          disabled={deviceSwitching || !device.available}
                          className={`flex-1 min-w-[80px] py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
                            isActive
                              ? `${colorMap[device.type] || 'bg-purple-600'} text-white`
                              : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                          } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                          <span className="flex items-center justify-center gap-2">
                            {device.type === 'cpu' ? (
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z"/></svg>
                            ) : (
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
                            )}
                            {labelMap[device.type] || device.id.toUpperCase()}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                  {devices.find(d => d.id === currentDevice) && (
                    <div className="mt-3 text-[10px] text-zinc-500">
                      {devices.find(d => d.id === currentDevice)?.name}
                    </div>
                  )}
                </div>

                {/* DRAM Extension */}
                <div className="flex items-center justify-between p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                      <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">DRAM Extension</div>
                      <div className="text-xs text-zinc-500">Use system RAM as VRAM fallback</div>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" checked={dramEnabled} onChange={(e) => setDramEnabled(e.target.checked)} className="sr-only peer" />
                    <div className="w-11 h-6 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-iris-accent" />
                  </label>
                </div>

                {/* DRAM Settings */}
                {dramEnabled && (
                  <div className="pl-4 border-l-2 border-iris-accent/30 space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="text-xs text-zinc-400">VRAM Threshold</label>
                        <span className="text-xs font-mono text-iris-accent">{vramThreshold} GB</span>
                      </div>
                      <input type="range" min={2} max={12} value={vramThreshold} step={1} onChange={(e) => setVramThreshold(Number(e.target.value))} className="w-full" />
                      <div className="text-[10px] text-zinc-600 mt-1">Enable DRAM extension when VRAM is below this threshold</div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="text-xs text-zinc-400">Max DRAM Usage</label>
                        <span className="text-xs font-mono text-iris-accent">{maxDram} GB</span>
                      </div>
                      <input type="range" min={4} max={64} value={maxDram} step={4} onChange={(e) => setMaxDram(Number(e.target.value))} className="w-full" />
                      <div className="text-[10px] text-zinc-600 mt-1">Maximum system RAM to use for model offloading</div>
                    </div>
                  </div>
                )}

                {/* NSFW Filter */}
                <div className="p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                        <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-white">NSFW Content Filter</div>
                        <div className="text-xs text-zinc-500">Block explicit content in prompts</div>
                      </div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input type="checkbox" checked={nsfwFilterEnabled} onChange={(e) => setNsfwFilterEnabled(e.target.checked)} className="sr-only peer" />
                      <div className="w-11 h-6 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-red-500" />
                    </label>
                  </div>
                  
                  {nsfwFilterEnabled && (
                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="text-xs text-zinc-400">Filter Strength</label>
                        <span className="text-xs font-mono text-iris-accent">{getNsfwStrengthLabel(nsfwStrength)}</span>
                      </div>
                      <input type="range" min={1} max={3} value={nsfwStrength} step={1} onChange={(e) => setNsfwStrength(Number(e.target.value))} className="w-full" />
                      <div className="flex justify-between text-[10px] text-zinc-600 mt-1">
                        <span>Relaxed</span>
                        <span>Standard</span>
                        <span>Strict</span>
                      </div>
                    </div>
                  )}
                  
                  {!nsfwFilterEnabled && (
                    <div className="flex items-center gap-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                      <svg className="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                      <span className="text-xs text-amber-300">Research Mode - Filter disabled for testing</span>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* Hardware Status */}
            <section className="glass-panel liquid-glass p-6 rounded-2xl">
              <h2 className="text-base font-bold text-white mb-5 flex items-center gap-2">
                <svg className="w-5 h-5 text-iris-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                Hardware Monitoring
              </h2>

              <div className="grid md:grid-cols-2 gap-4">
                {/* GPU Usage */}
                <div className="p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="text-[10px] text-zinc-400 uppercase font-semibold mb-1 tracking-wider">GPU Usage</div>
                  <div className="flex items-end justify-between">
                    <div className="text-2xl font-bold text-white font-mono">{Math.round(gpuLoad)}%</div>
                    <div className="text-sm text-iris-accentLight font-mono">{gpuTemp > 0 ? `${Math.round(gpuTemp)}°C` : '--°C'}</div>
                  </div>
                  <div className="w-full bg-black/30 h-2 rounded-full mt-3 overflow-hidden border border-white/5">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(gpuLoad, 100)}%`, background: gpuLoad > 80 ? 'linear-gradient(90deg, #ef4444, #f87171)' : gpuLoad > 50 ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' : 'linear-gradient(90deg, #8b5cf6, #a78bfa)' }} />
                  </div>
                  <div className="flex justify-between mt-2 text-[10px] text-zinc-500">
                    <span>{powerDraw > 0 ? `${powerDraw.toFixed(0)} W` : '-- W'}</span>
                    <span>{gpuName.replace('NVIDIA ', '').replace('GeForce ', '')}</span>
                  </div>
                </div>

                {/* VRAM Usage */}
                <div className="p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="text-[10px] text-zinc-400 uppercase font-semibold mb-1 tracking-wider">VRAM Usage</div>
                  <div className="flex items-end justify-between">
                    <div className="text-2xl font-bold text-white font-mono">{vramUsed.toFixed(1)} GB</div>
                    <div className="text-sm text-zinc-500">/ {vramTotal.toFixed(1)} GB</div>
                  </div>
                  <div className="w-full bg-black/30 h-2 rounded-full mt-3 overflow-hidden border border-white/5">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${vramPerc}%`, background: vramPerc > 85 ? 'linear-gradient(90deg, #ef4444, #f87171)' : vramPerc > 60 ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' : 'linear-gradient(90deg, #6366f1, #818cf8)' }} />
                  </div>
                </div>

                {/* CPU Usage */}
                <div className="p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="text-[10px] text-zinc-400 uppercase font-semibold mb-1 tracking-wider">CPU Usage</div>
                  <div className="flex items-end justify-between">
                    <div className="text-2xl font-bold text-white font-mono">{Math.round(cpuPercent)}%</div>
                    <div className="text-sm text-zinc-500 font-mono">{cpuFreq > 0 ? `${cpuFreq.toFixed(2)} GHz` : '-- GHz'}</div>
                  </div>
                  <div className="w-full bg-black/30 h-2 rounded-full mt-3 overflow-hidden border border-white/5">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(cpuPercent, 100)}%`, background: cpuPercent > 80 ? 'linear-gradient(90deg, #ef4444, #f87171)' : cpuPercent > 50 ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' : 'linear-gradient(90deg, #10b981, #34d399)' }} />
                  </div>
                  <div className="flex justify-between mt-2 text-[10px] text-zinc-500">
                    <span>{cpuCores > 0 ? `${cpuCores} Cores` : '-- Cores'}</span>
                  </div>
                </div>

                {/* RAM Usage */}
                <div className="p-4 bg-iris-card rounded-xl border border-iris-border">
                  <div className="text-[10px] text-zinc-400 uppercase font-semibold mb-1 tracking-wider">RAM Usage</div>
                  <div className="flex items-end justify-between">
                    <div className="text-2xl font-bold text-white font-mono">{ramUsed.toFixed(1)} GB</div>
                    <div className="text-sm text-zinc-500">/ {ramTotal.toFixed(1)} GB</div>
                  </div>
                  <div className="w-full bg-black/30 h-2 rounded-full mt-3 overflow-hidden border border-white/5">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${ramPercent}%`, background: ramPercent > 85 ? 'linear-gradient(90deg, #ef4444, #f87171)' : ramPercent > 60 ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' : 'linear-gradient(90deg, #f59e0b, #fbbf24)' }} />
                  </div>
                </div>
              </div>
            </section>

            {/* System Info */}
            <section className="glass-panel liquid-glass p-6 rounded-2xl">
              <h2 className="text-base font-bold text-white mb-5">Software Stack</h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2.5 border-b border-iris-border">
                  <span className="text-zinc-500 text-sm">Operating System</span>
                  <span className="text-white font-medium text-sm">{versionInfo?.os || 'Loading...'}</span>
                </div>
                <div className="flex justify-between items-center py-2.5 border-b border-iris-border">
                  <span className="text-zinc-500 text-sm">Python Version</span>
                  <span className="text-white font-medium font-mono text-sm">{versionInfo?.python_version || '--'}</span>
                </div>
                <div className="flex justify-between items-center py-2.5 border-b border-iris-border">
                  <span className="text-zinc-500 text-sm">PyTorch / CUDA</span>
                  <span className="text-white font-medium font-mono text-sm">{versionInfo ? `${versionInfo.pytorch_version} / ${versionInfo.cuda_version}` : '--'}</span>
                </div>
                <div className="flex justify-between items-center py-2.5">
                  <span className="text-zinc-500 text-sm">I.R.I.S. Core</span>
                  <span className="text-iris-accentLight font-bold font-mono text-sm">v1.2.0</span>
                </div>
              </div>
            </section>

            {/* Discord Integration */}
            <section className="glass-panel liquid-glass p-6 rounded-2xl">
              <h2 className="text-base font-bold text-white mb-5 flex items-center gap-2">
                <svg className="w-5 h-5 text-indigo-400" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" /></svg>
                Discord Integration
              </h2>

              <div className="flex items-center justify-between p-4 bg-iris-card rounded-xl border border-iris-border">
                <div className="flex items-center gap-3">
                  <div>
                    <div className="text-sm font-medium text-white">Discord Bot</div>
                    <div className="text-xs text-zinc-500">Share images to Discord</div>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" checked={discordEnabled} onChange={(e) => handleDiscordToggle(e.target.checked)} className="sr-only peer" />
                  <div className="w-11 h-6 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-500" />
                </label>
              </div>
              <div className="flex items-center gap-2 p-3 mt-3 bg-iris-bg rounded-lg">
                <span className={`w-2 h-2 rounded-full ${
                  discordStatus === 'Running' ? 'bg-green-500' :
                  discordStatus === 'Starting...' || discordStatus === 'Stopping...' ? 'bg-yellow-500 animate-pulse' :
                  discordStatus === 'Token missing' || discordStatus === 'Failed' || discordStatus === 'Server offline' ? 'bg-red-500' :
                  'bg-amber-500 animate-pulse'
                }`} />
                <span className={`text-xs ${
                  discordStatus === 'Running' ? 'text-green-400' :
                  discordStatus === 'Starting...' || discordStatus === 'Stopping...' ? 'text-yellow-400' :
                  discordStatus === 'Token missing' || discordStatus === 'Failed' || discordStatus === 'Server offline' ? 'text-red-400' :
                  'text-zinc-400'
                }`}>{discordStatus}</span>
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  )
}
