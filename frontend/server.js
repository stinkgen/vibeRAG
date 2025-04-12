const express = require('express');
const path = require('path');
// const { createProxyMiddleware } = require('http-proxy-middleware'); // Remove HPM
const WebSocket = require('ws'); // Import the ws library
const axios = require('axios'); // Add axios for manual HTTP proxying
const url = require('url'); // Import the url module

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

// === Body Parsing Middleware (BEFORE Proxy) ===
// Handles application/json
app.use(express.json());
// Handles application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: false }));
// ==============================================

const port = process.env.PORT || 3000; // Use environment variable or default to 3000
const backendTarget = 'http://backend:8000'; // Target backend service
const backendWsTarget = 'ws://backend:8000/api/v1/ws/chat'; // WebSocket specific target

// === NEW Manual HTTP Proxy using Axios ===
app.use('/api', async (req, res) => {
  const backendUrl = `${backendTarget}${req.originalUrl}`;
  console.log(`[Manual Proxy] Forwarding ${req.method} ${req.originalUrl} to ${backendUrl}`);

  const contentType = req.headers['content-type'] || '';
  // Determine if we should stream the request body or send parsed body
  const shouldStream = req.method !== 'GET' && req.method !== 'HEAD' && !contentType.includes('application/json') && !contentType.includes('application/x-www-form-urlencoded');

  console.log(`[Manual Proxy] Content-Type: ${contentType}, ShouldStream: ${shouldStream}`);

  try {
    // Prepare headers, ensuring original Content-Type is passed
    const requestHeaders = {
      ...req.headers,
      host: new URL(backendTarget).host, // Set correct host for backend
      connection: 'keep-alive',
      // Let axios handle content-length for non-streamed requests
      'content-length': shouldStream ? req.headers['content-length'] : undefined,
      'transfer-encoding': undefined, // Avoid potential issues
    };

    const axiosConfig = {
      method: req.method,
      url: backendUrl,
      headers: requestHeaders,
      // Send parsed body for JSON/form data, otherwise stream
      data: shouldStream ? req : req.body,
      // Set responseType based on whether we are streaming
      responseType: shouldStream ? 'stream' : 'json', // Expect JSON if not streaming
      validateStatus: status => true // Handle all status codes manually
    };

    const response = await axios(axiosConfig);

    // Forward status code and headers from backend response
    res.status(response.status);
    // Filter out headers that shouldn't be forwarded directly
    const responseHeaders = { ...response.headers };
    delete responseHeaders['transfer-encoding'];
    delete responseHeaders['connection'];
    // delete responseHeaders['content-length']; // Let express handle this
    res.set(responseHeaders);

    if (shouldStream && response.data?.pipe) {
      // Pipe the backend response stream to the client response
      response.data.pipe(res);
    } else {
      // Send the parsed JSON response
      res.json(response.data);
    }

  } catch (error) {
    console.error(`[Manual Proxy] Error forwarding request to ${backendUrl}:`, error.message);
    if (error.response) {
      // If backend responded with an error
      console.error(`[Manual Proxy] Backend error status: ${error.response.status}`);
      res.status(error.response.status).send(error.response.data);
    } else if (error.request) {
      // If request was made but no response received (e.g., timeout, ECONNREFUSED)
      console.error(`[Manual Proxy] No response from backend (ECONNREFUSED or timeout).`);
      res.status(504).send('Gateway Timeout - Backend did not respond');
    } else {
      // Other errors
      console.error(`[Manual Proxy] Non-response error: ${error.message}`);
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
  // Parse the request URL to separate pathname and query string
  const parsedUrl = url.parse(request.url);
  const pathname = parsedUrl.pathname;

  console.log(`[Upgrade] Attempting upgrade for path: ${request.url} (pathname: ${pathname})`);

  // Check only the pathname for the WebSocket route
  if (pathname === '/api/v1/ws/chat') {
    console.log(`[Upgrade] Pathname matches. Handling upgrade...`);

    wss.handleUpgrade(request, clientSocket, head, (wsClient) => {
      console.log('[WSS] Client WebSocket handshake complete.');
      setupHeartbeat(wsClient, 'Client'); // Setup client heartbeat

      // Construct backend URL - Forward query string (like ?token=...)
      const backendWsTargetUrl = `${backendWsTarget}${parsedUrl.search || ''}`;
      console.log(`[WSS] Establishing backend connection to ${backendWsTargetUrl}`);

      const backendSocket = new WebSocket(backendWsTargetUrl);
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
          // Log the type and potentially the content (truncated)
          const messageType = Buffer.isBuffer(message) ? 'Buffer' : (typeof message);
          const messagePreview = Buffer.isBuffer(message) ? message.toString('utf8', 0, 100) : (typeof message === 'string' ? message.substring(0, 100) : '');
          console.log(`[WSS Relay C->B] Received message from client (Type: ${messageType}). Preview: ${messagePreview}...`);
          
          if (backendSocket.readyState === WebSocket.OPEN) {
            // Ensure message is sent as a string to the backend
            const messageString = Buffer.isBuffer(message) ? message.toString('utf8') : message;
            backendSocket.send(messageString); 
          }
        });

        // --- Relay messages from Backend to Client ---
        backendSocket.on('message', (message) => {
          const messageString = message.toString('utf8');
          console.log('[WSS Relay B->C] Received message from backend (forwarding as string)');
          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.send(messageString);
          }
        });

        // --- Handle Close Events ---
        wsClient.on('close', (code, reason) => {
          console.log(`[WSS Client] Closed connection. Code: ${code}, Reason: ${reason?.toString()}`);
          if (backendSocket.readyState === WebSocket.OPEN) {
            backendSocket.close();
          }
        });

        backendSocket.on('close', (code, reason) => {
          console.log(`[WSS Backend] Closed connection. Code: ${code}, Reason: ${reason?.toString()}`);
          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.close();
          }
        });

        // --- Handle Error Events during active connection ---
        wsClient.on('error', (err) => {
          console.error(`[WSS Client] Error: ${err.message}`);
          if (backendSocket.readyState === WebSocket.OPEN) {
            backendSocket.close(1011, 'Client error');
          }
        });

         backendSocket.on('error', (err) => {
          console.error(`[WSS Backend] Error during relay: ${err.message}`);
          if (wsClient.readyState === WebSocket.OPEN) {
            wsClient.close(1011, 'Backend error');
          }
        });
      });
      // Note: Initial backend connection error handled by the 'error' handler above

    }); // End of wss.handleUpgrade callback

  } else {
    // If the path doesn't match, destroy the socket
    console.log(`[Upgrade] Pathname ${pathname} does not match target /api/v1/ws/chat. Destroying socket.`);
    clientSocket.destroy();
  }
}); // End of server.on('upgrade')

console.log("Express server configured with manual WebSocket proxying using 'ws' library."); 