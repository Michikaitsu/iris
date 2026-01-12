/**
 * I.R.I.S. WebSocket Manager
 * Handles automatic reconnection and message queuing
 */

class WebSocketManager {
    constructor(url, options = {}) {
        this.url = url;
        this.options = {
            reconnectInterval: 1000,
            maxReconnectInterval: 30000,
            reconnectDecay: 1.5,
            maxReconnectAttempts: 10,
            ...options
        };
        
        this.ws = null;
        this.reconnectAttempts = 0;
        this.messageQueue = [];
        this.listeners = new Map();
        this.isIntentionallyClosed = false;
        
        this.connect();
    }
    
    /**
     * Establish WebSocket connection
     */
    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }
        
        console.log(`[WS] Connecting to ${this.url}...`);
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('[WS] Connected');
            this.reconnectAttempts = 0;
            this.flushMessageQueue();
            this.emit('open');
        };
        
        this.ws.onclose = (event) => {
            console.log(`[WS] Disconnected (code: ${event.code})`);
            this.emit('close', event);
            
            if (!this.isIntentionallyClosed) {
                this.scheduleReconnect();
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('[WS] Error:', error);
            this.emit('error', error);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.emit('message', data);
                
                // Emit typed events based on message type
                if (data.type) {
                    this.emit(data.type, data);
                }
            } catch (e) {
                this.emit('message', event.data);
            }
        };
    }
    
    /**
     * Schedule reconnection with exponential backoff
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('[WS] Max reconnection attempts reached');
            this.emit('maxReconnectAttempts');
            return;
        }
        
        const delay = Math.min(
            this.options.reconnectInterval * Math.pow(this.options.reconnectDecay, this.reconnectAttempts),
            this.options.maxReconnectInterval
        );
        
        this.reconnectAttempts++;
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        this.emit('reconnecting', { attempt: this.reconnectAttempts, delay });
        
        setTimeout(() => this.connect(), delay);
    }
    
    /**
     * Send message (queues if not connected)
     */
    send(data) {
        const message = typeof data === 'string' ? data : JSON.stringify(data);
        
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(message);
        } else {
            console.log('[WS] Queuing message (not connected)');
            this.messageQueue.push(message);
        }
    }
    
    /**
     * Flush queued messages after reconnection
     */
    flushMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.ws.send(message);
        }
    }
    
    /**
     * Add event listener
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
        return this;
    }
    
    /**
     * Remove event listener
     */
    off(event, callback) {
        if (!this.listeners.has(event)) return;
        
        const callbacks = this.listeners.get(event);
        const index = callbacks.indexOf(callback);
        if (index > -1) {
            callbacks.splice(index, 1);
        }
        return this;
    }
    
    /**
     * Emit event to listeners
     */
    emit(event, data) {
        if (!this.listeners.has(event)) return;
        
        this.listeners.get(event).forEach(callback => {
            try {
                callback(data);
            } catch (e) {
                console.error(`[WS] Error in ${event} handler:`, e);
            }
        });
    }
    
    /**
     * Close connection
     */
    close() {
        this.isIntentionallyClosed = true;
        if (this.ws) {
            this.ws.close();
        }
    }
    
    /**
     * Get connection state
     */
    get state() {
        if (!this.ws) return 'CLOSED';
        return ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][this.ws.readyState];
    }
    
    /**
     * Check if connected
     */
    get isConnected() {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

// Export for use in other scripts
window.WebSocketManager = WebSocketManager;
