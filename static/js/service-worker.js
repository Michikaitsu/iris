/**
 * I.R.I.S. Service Worker
 * Provides offline support and caching for the application
 */

const CACHE_NAME = 'iris-cache-v1';
const STATIC_ASSETS = [
    '/',
    '/generate',
    '/gallery',
    '/settings',
    '/static/css/generate.css',
    '/static/css/gallery.css',
    '/static/css/settings.css',
    '/static/css/index.css',
    '/static/js/scripts.js',
    '/static/js/websocket-manager.js',
    '/assets/fav.ico'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[ServiceWorker] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[ServiceWorker] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting())
            .catch((err) => console.error('[ServiceWorker] Cache failed:', err))
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[ServiceWorker] Activating...');
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== CACHE_NAME)
                        .map((name) => {
                            console.log('[ServiceWorker] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => self.clients.claim())
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip WebSocket and API requests
    if (url.pathname.startsWith('/ws/') || 
        url.pathname.startsWith('/api/') ||
        request.method !== 'GET') {
        return;
    }

    // Cache-first for static assets
    if (url.pathname.startsWith('/static/') || 
        url.pathname.startsWith('/assets/')) {
        event.respondWith(cacheFirst(request));
        return;
    }

    // Network-first for HTML pages
    event.respondWith(networkFirst(request));
});

/**
 * Cache-first strategy
 */
async function cacheFirst(request) {
    const cached = await caches.match(request);
    if (cached) {
        return cached;
    }
    
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
        }
        return response;
    } catch (err) {
        console.error('[ServiceWorker] Fetch failed:', err);
        return new Response('Offline', { status: 503 });
    }
}

/**
 * Network-first strategy
 */
async function networkFirst(request) {
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
        }
        return response;
    } catch (err) {
        const cached = await caches.match(request);
        if (cached) {
            return cached;
        }
        return new Response(offlineHTML(), {
            status: 503,
            headers: { 'Content-Type': 'text/html' }
        });
    }
}

/**
 * Offline fallback page
 */
function offlineHTML() {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>I.R.I.S. - Offline</title>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: #0a0a0f;
            color: #f4f4f5;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            padding: 2rem;
        }
        h1 { color: #8b5cf6; }
        p { color: #71717a; }
        button {
            background: #8b5cf6;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            margin-top: 1rem;
        }
        button:hover { background: #7c3aed; }
    </style>
</head>
<body>
    <div class="container">
        <h1>You're Offline</h1>
        <p>I.R.I.S. requires a connection to generate images.</p>
        <button onclick="location.reload()">Try Again</button>
    </div>
</body>
</html>`;
}
