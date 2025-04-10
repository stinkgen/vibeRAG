# Stage 1: Build the React application
FROM node:20-alpine AS build

WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application source code
COPY . .

# Build the React app
RUN npm run build

# Stage 2: Create the final image using a lighter base image
FROM node:20-alpine

WORKDIR /app

# Copy necessary files from build stage
COPY --from=build /app/build ./build
COPY --from=build /app/package.json .
COPY --from=build /app/server.js .

# Install only production dependencies
# Use --omit=dev if using npm v7+, otherwise use --production
# Check npm version or adjust accordingly
RUN npm install --omit=dev 

# Expose the port the app runs on (defined in server.js or default 3000)
EXPOSE 3000

# Define the command to run the application
CMD ["node", "server.js"] 