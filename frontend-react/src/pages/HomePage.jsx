import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'
import Footer from '../components/Footer'
import { getOutputGallery } from '../lib/api'

export default function HomePage() {
  const [totalGenerated, setTotalGenerated] = useState(0)

  useEffect(() => {
    getOutputGallery()
      .then((data) => {
        const count = data.images?.length || 0
        let current = 0
        const increment = Math.ceil(count / 20)
        const timer = setInterval(() => {
          current += increment
          if (current >= count) {
            current = count
            clearInterval(timer)
          }
          setTotalGenerated(current)
        }, 50)
        return () => clearInterval(timer)
      })
      .catch(() => setTotalGenerated(0))
  }, [])

  return (
    <div className="relative min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 container mx-auto px-6 flex flex-col items-center justify-center relative z-10 text-center py-16">
        <div className="hero-glow" />

        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full liquid-glass text-xs font-medium text-iris-accentLight mb-8 animate-fade-in-up font-mono">
          <span className="w-2 h-2 rounded-full bg-iris-accent animate-pulse" />
          v1.1.0 Stable
        </div>

        <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight max-w-4xl mx-auto leading-tight">
          Create Stunning <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-indigo-400 animate-gradient">
            AI Art Locally
          </span>
        </h1>

        <p className="text-lg text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed font-light">
          Intelligent Rendering & Image Synthesis. Unleash your creativity with a powerful, free, and private AI generation suite running directly on your hardware.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 w-full justify-center">
          <Link
            to="/generate"
            className="btn-primary px-8 py-4 rounded-xl text-lg font-bold flex items-center justify-center gap-2 group shadow-lg shadow-purple-900/40 text-white transition-all"
          >
            <span>Start Creating</span>
            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
          <Link
            to="/gallery"
            className="btn-secondary px-8 py-4 rounded-xl text-lg font-medium flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span>View Gallery</span>
          </Link>
        </div>

        {/* Stats */}
        <div className="mt-16 glass-panel p-8 w-full max-w-5xl grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="text-4xl font-bold stat-value mb-1 font-mono">{totalGenerated}</div>
            <div className="text-xs text-zinc-500 uppercase tracking-wide font-semibold">Images Created</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold stat-value mb-1 font-mono">12+</div>
            <div className="text-xs text-zinc-500 uppercase tracking-wide font-semibold">Art Models</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold stat-value mb-1 font-mono">4x</div>
            <div className="text-xs text-zinc-500 uppercase tracking-wide font-semibold">AI Upscaling</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold stat-value mb-1 font-mono">100%</div>
            <div className="text-xs text-zinc-500 uppercase tracking-wide font-semibold">Offline & Free</div>
          </div>
        </div>
      </main>

      {/* Features */}
      <section className="container mx-auto px-6 py-20 relative z-10">
        <h2 className="text-2xl font-bold mb-10 text-center text-white">Power at your fingertips</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          <FeatureCard
            icon={<PaletteIcon />}
            title="Fine-Tuned Models"
            description="Switch between Anime, Realistic, and Abstract styles with optimized presets."
          />
          <FeatureCard
            icon={<BoltIcon />}
            title="Lightning Fast"
            description="Optimized pipeline for quick iterations and real-time prompt refinement."
          />
          <FeatureCard
            icon={<ExpandIcon />}
            title="Upscaling"
            description="Enhance details up to 4x using advanced neural network upscalers."
            badge="New"
          />
          <FeatureCard
            icon={<SlidersIcon />}
            title="Full Control"
            description="Fine-tune Seed, CFG Scale, Steps, and Samplers for perfect results."
          />
        </div>
      </section>

      <Footer />
    </div>
  )
}

function FeatureCard({ icon, title, description, badge }) {
  return (
    <div className="feature-card">
      <div className="icon-box">{icon}</div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-bold text-white">{title}</h3>
        {badge && (
          <span className="text-[10px] bg-iris-accent/20 text-iris-accentLight px-2 py-0.5 rounded border border-iris-accent/30 font-mono">
            {badge}
          </span>
        )}
      </div>
      <p className="text-sm text-zinc-400 leading-relaxed">{description}</p>
    </div>
  )
}

function PaletteIcon() {
  return (
    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
    </svg>
  )
}

function BoltIcon() {
  return (
    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  )
}

function ExpandIcon() {
  return (
    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
    </svg>
  )
}

function SlidersIcon() {
  return (
    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
    </svg>
  )
}
