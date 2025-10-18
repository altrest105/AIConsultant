FROM node:20-alpine AS dependency_builder

WORKDIR /app
COPY package*.json ./

RUN npm install

FROM node:20-alpine AS development

WORKDIR /app

COPY --from=dependency_builder /app/node_modules ./node_modules
COPY package*.json ./

COPY . .

EXPOSE 5173

ENV NODE_ENV=development

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
