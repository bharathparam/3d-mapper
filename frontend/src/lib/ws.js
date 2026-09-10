/**
 * WebSocket client for real-time job progress.
 *
 * Usage:
 *   const sub = subscribeToJob(jobId, (event) => { ... })
 *   // later:
 *   sub.close()
 */
export function subscribeToJob(jobId, onEvent) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.hostname
  const port = import.meta.env.DEV ? '8000' : window.location.port
  const url = `${protocol}://${host}:${port}/ws/jobs/${jobId}`

  const ws = new WebSocket(url)

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      onEvent(data)
    } catch (err) {
      console.warn('[ws] parse error', err)
    }
  }

  ws.onerror = (e) => {
    console.warn('[ws] error', e)
    onEvent({ event: 'ws_error', message: 'WebSocket connection error' })
  }

  ws.onclose = () => {
    onEvent({ event: 'ws_closed' })
  }

  // Keep-alive ping every 20 s
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping')
  }, 20000)

  return {
    close() {
      clearInterval(pingInterval)
      ws.close()
    },
  }
}
