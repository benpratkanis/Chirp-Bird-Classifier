FROM node:20-alpine

WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

RUN npm install

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 5173

# Command to run the application
# --host is needed to expose the app to the host machine
CMD ["npm", "run", "dev", "--", "--host"]
