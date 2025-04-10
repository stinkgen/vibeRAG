const express = require('express');
const path = require('path');
// const { createProxyMiddleware } = require('http-proxy-middleware'); // Remove HPM
const WebSocket = require('ws'); // Import the ws library
const axios = require('axios'); // Add axios for manual HTTP proxying

// === SERVER STARTUP LOG ===
console.log("*** Server.js starting - MANUAL HTTP & WS Proxying ***"); 
// ==========================

const app = express();

// === VERY FIRST MIDDLEWARE: Log ALL incoming requests ===
app.use((req, res, next) => {
  console.log(`>>> REQUEST RECEIVED: ${req.method} ${req.originalUrl}`);
  next(); // Pass control to the next middleware/route handler
});
// ======================================================

const port = process.env.PORT || 3000; // Use environment variable or default to 3000
const backendTarget = 'http://backend:8000'; // Target backend service
const backendWsTarget = 'ws://backend:8000/api/v1/ws/chat'; // WebSocket specific target

// === NEW Manual HTTP Proxy using Axios ===
app.use('/api', async (req, res) => {
  const backendUrl = `${backendTarget}${req.originalUrl}`;
  console.log(`[Manual Proxy] Forwarding ${req.method} ${req.originalUrl} to ${backendUrl}`);

  try {
    // Prepare headers, ensuring original Content-Type is passed for multipart
    const requestHeaders = {
      ...req.headers,
      host: new URL(backendTarget).host, // Set correct host for backend
      // Remove headers that might cause issues or are set by axios/http agent
      'content-length': undefined, 
      'transfer-encoding': undefined,
      connection: 'keep-alive',
    };
    // Ensure Content-Type from the original request is preserved
    if (req.headers['content-type']) {
        requestHeaders['content-type'] = req.headers['content-type'];
    }

    const response = await axios({
      method: req.method,
      url: backendUrl,
      headers: requestHeaders, // Use the prepared headers
      // Stream the incoming request body directly
      // Axios should handle piping req when it's a stream
      data: req, 
      responseType: 'stream', 
      validateStatus: status => true 
    });

    // Forward status code and headers from backend response
    res.writeHead(response.status, response.headers);
    // Pipe the backend response stream to the client response
    response.data.pipe(res);

  } catch (error) {
    console.error(`[Manual Proxy] Error forwarding request to ${backendUrl}:`, error.message);
    if (error.response) {
      // If backend responded with an error
      res.writeHead(error.response.status, error.response.headers);
      error.response.data.pipe(res);
    } else if (error.request) {
      // If request was made but no response received (e.g., timeout, ECONNREFUSED)
      res.status(504).send('Gateway Timeout - Backend did not respond');
    } else {
      // Other errors
      res.status(500).send('Internal Server Error during proxying');
    }
  }
});

// Serve static files from the React app build directory (Applied AFTER proxy)
app.use(express.static(path.join(__dirname, 'build')));

// Handles any requests that don't match the ones above (e.g., for client-side routing)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// Start the HTTP server
const server = app.listen(port, () => {
  console.log(`Frontend Express server listening on port ${port}`);
  console.log(`Proxying HTTP API requests (excluding WS) to ${backendTarget}`);
  console.log(`Manually proxying WebSocket requests for /api/v1/ws/chat to ${backendWsTarget}`);
});

// Create a WebSocket server instance without attaching it to a specific HTTP server
const wss = new WebSocket.WebSocketServer({ noServer: true });

// === WebSocket Proxy Logic ===

// Heartbeat interval (milliseconds)
const HEARTBEAT_INTERVAL = 30000; // 30 seconds
const HEARTBEAT_TIMEOUT = 60000; // 60 seconds

// Function to manage heartbeat for a WebSocket connection
function setupHeartbeat(ws, name) {
    let isAlive = true;
    let pingTimeout = null;
    let interval = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            if(interval) clearInterval(interval);
            return;
        }
        
        isAlive = false; 
        ws.ping(() => {}); 

        pingTimeout = setTimeout(() => {
            if (!isAlive) {
                console.warn(`[Heartbeat ${name}] No pong received. Terminating connection.`);
                ws.terminate(); 
                if(interval) clearInterval(interval); 
            }
        }, HEARTBEAT_TIMEOUT);

    }, HEARTBEAT_INTERVAL);

    ws.on('pong', () => {
        isAlive = true;
        if (pingTimeout) clearTimeout(pingTimeout); 
    });

    ws.on('close', () => {
        console.log(`[Heartbeat ${name}] Connection closed, clearing interval.`);
        if (interval) clearInterval(interval);
        if (pingTimeout) clearTimeout(pingTimeout);
    });
    
    ws.on('error', (err) => {
        console.error(`[Heartbeat ${name}] Error on connection: ${err.message}. Clearing interval.`);
        if (interval) clearInterval(interval);
        if (pingTimeout) clearTimeout(pingTimeout);
    });
}

// Handle WebSocket upgrades manually using the 'ws' library
server.on('upgrade', (request, clientSocket, head) => {
  const pathname = request.url;
  console.log(`[Upgrade] Attempting upgrade for path: ${pathname}`);

  if (pathname === '/api/v1/ws/chat') {
    console.log(`[Upgrade] Path matches. Handling upgrade...`);

    wss.handleUpgrade(request, clientSocket, head, (wsClient) => {
      console.log('[WSS] Client WebSocket handshake complete.');
      setupHeartbeat(wsClient, 'Client'); // Setup client heartbeat

      console.log(`[WSS] Establishing backend connection to ${backendWsTarget}`);
      const backendSocket = new WebSocket(backendWsTarget);
      setupHeartbeat(backendSocket, 'Backend'); // Setup backend heartbeat

      // --- Error Handling for Backend Connection ---
      backendSocket.on('error', (err) => {
        console.error(`[WSS Backend] Connection error: ${err.message}`);
        wsClient.close(1011, 'Backend connection error'); // Close client if backend fails
      });

      backendSocket.on('open', () => {
        console.log('[WSS Backend] Connection established.');

        // --- Relay messages from Client to Backend ---
        wsClient.on('message', (message) => {
          console.log('[WSS Relay C->B] Received message from client');
          if (backendSocket.readyState === WebSocket.OPEN) {
            backendSocket.send(message);
          }
        });

        // --- Relay messages from Backend to Client ---
        backendSocket.on('message', (message) => {
          // console.log('[WSS Relay B->C] Raw message from backend:', message); // Debug log
          
          // Convert message buffer to UTF-8 string before sending to client
          const messageString = message.toString('utf8');
          // console.log('[WSS Relay B->C] Forwarding string to client:', messageString);
          
          console.log('[WSS Relay B->C] Received message from backend (forwarding as string)'); // Keep original log concise
          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.send(messageString); // Send the string, not the raw buffer
          }
        });

        // --- Handle Close Events ---
        wsClient.on('close', (code, reason) => {
          console.log(`[WSS Client] Closed connection. Code: ${code}, Reason: ${reason?.toString()}`);
          if (backendSocket.readyState === WebSocket.OPEN) {
            backendSocket.close(); // Close backend when client closes
          }
        });

        backendSocket.on('close', (code, reason) => {
          console.log(`[WSS Backend] Closed connection. Code: ${code}, Reason: ${reason?.toString()}`);
          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.close(); // Close client when backend closes
          }
        });

        // --- Handle Error Events during active connection ---
        wsClient.on('error', (err) => {
          console.error(`[WSS Client] Error: ${err.message}`);
          if (backendSocket.readyState === WebSocket.OPEN) {
            backendSocket.close(1011, 'Client error');
          }
        });

         backendSocket.on('error', (err) => { // Note: This duplicates the outer error handler, but scoped here too
          console.error(`[WSS Backend] Error during relay: ${err.message}`);
          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.close(1011, 'Backend error');
          }
        });
      });
      
      // Initial backend connection error is handled outside/above the 'open' handler

    }); // End of wss.handleUpgrade callback

  } else {
    // If the path doesn't match, destroy the socket to prevent hanging connections
    console.log(`[Upgrade] Path ${pathname} does not match. Destroying socket.`);
    clientSocket.destroy();
  }
});

console.log('Express server configured with manual WebSocket proxying using \'ws\' library.'); 