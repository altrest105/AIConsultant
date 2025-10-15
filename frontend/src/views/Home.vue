<template>
  <div class="home-hero">
    <!-- Background -->
    <div class="cyber-background">
      <div class="grid-lines"></div>
      <div class="floating-particles">
        <div class="particle" v-for="i in 30" :key="i" :style="getParticleStyle(i)"></div>
      </div>
      <div class="gradient-orb orb-1"></div>
      <div class="gradient-orb orb-2"></div>
      <div class="gradient-orb orb-3"></div>
    </div>

    <!-- Content -->
    <div class="hero-content animate-fadeInUp">
      <h1 class="hero-title">
        <span class="title-gradient text-gradient">Цифровой Консультант</span>
        <span class="company-name">ПАО «Транснефть»</span>
      </h1>
      
      <p class="hero-subtitle">
        Интеллектуальный AI помощник нового поколения
      </p>

      <div class="features-grid">
        <div class="feature-card" v-for="(feature, index) in features" :key="index" :style="{ animationDelay: `${0.8 + index * 0.1}s` }">
          <div class="feature-icon">{{ feature.icon }}</div>
          <div class="feature-text">{{ feature.text }}</div>
        </div>
      </div>

      <RouterLink to="/chat" class="start-button">
        <span class="button-text">Начать консультацию</span>
        <span class="button-arrow">→</span>
      </RouterLink>

    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const features = ref([
  { icon: '🤖', text: 'Поддержка ИИ' },
  { icon: '💬', text: 'Работа 24/7' },
  { icon: '⚡', text: 'Быстрые ответы' }
]);

function getParticleStyle(index) {
  return {
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    animationDelay: `${Math.random() * 5}s`,
    animationDuration: `${10 + Math.random() * 10}s`
  };
}
</script>

<style scoped>
.home-hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 50%, #e3f2fd 100%);
  overflow: hidden;
  padding: 40px 20px;
}

.cyber-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

.grid-lines {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    linear-gradient(rgba(0, 102, 204, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 102, 204, 0.05) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
  opacity: 0.5;
}

@keyframes gridMove {
  0% { transform: translate(0, 0); }
  100% { transform: translate(50px, 50px); }
}

.floating-particles {
  position: absolute;
  width: 100%;
  height: 100%;
}

.particle {
  position: absolute;
  width: 3px;
  height: 3px;
  background: rgba(0, 102, 204, 0.4);
  border-radius: 50%;
  box-shadow: 0 0 10px rgba(0, 102, 204, 0.3);
  animation: floatParticle 15s infinite ease-in-out;
}

@keyframes floatParticle {
  0%, 100% {
    transform: translateY(0) translateX(0);
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    transform: translateY(-100vh) translateX(30px);
    opacity: 0;
  }
}

.gradient-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(120px);
  opacity: 0.1;
  animation: orbFloat 20s infinite ease-in-out;
}

.text-gradient {
  background: linear-gradient(90deg, var(--transneft-blue), var(--transneft-red));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

@keyframes orbFloat {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  33% {
    transform: translate(50px, -50px) scale(1.1);
  }
  66% {
    transform: translate(-30px, 30px) scale(0.9);
  }
}

.hero-content {
  position: relative;
  z-index: 1;
  text-align: center;
  max-width: 800px;
}

.logo-section {
  margin-bottom: 40px;
  animation: fadeInScale 1s ease-out 0.2s both;
}

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.8);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.logo-circle {
  width: 150px;
  height: 150px;
  margin: 0 auto;
  background: #ffffff;
  border-radius: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 32px rgba(0, 30, 60, 0.15);
  position: relative;
}

.logo-icon {
  width: 130px;
  height: 130px;
  display: block;
  margin: 0 auto;
  object-fit: contain;
}

.logo-circle::after {
  content: '';
  position: absolute;
  top: -5px;
  right: -5px;
  width: 20px;
  height: 20px;
  background: var(--transneft-red);
  border-radius: 50%;
  box-shadow: 0 0 20px var(--transneft-red);
  animation: pulse 2s ease-in-out infinite;
}

.hero-title {
  margin: 0 0 20px 0;
  font-size: 48px;
  font-weight: 800;
  line-height: 1.2;
  animation: fadeInUp 1s ease-out 0.4s both;
}

.title-gradient {
  display: block;
  margin-bottom: 10px;
}

.company-name {
  display: block;
  color: var(--transneft-darker);
  font-size: 42px;
}

.hero-subtitle {
  color: var(--transneft-blue);
  font-size: 20px;
  margin: 0 0 50px 0;
  font-weight: 500;
  letter-spacing: 0.5px;
  animation: fadeInUp 1s ease-out 0.6s both;
}

.features-grid {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-bottom: 50px;
  flex-wrap: wrap;
}

.feature-card {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(0, 102, 204, 0.15);
  border-radius: var(--radius-lg);
  padding: 20px 30px;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: all var(--transition-normal);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
  animation: fadeInUp 1s ease-out both;
  position: relative;
  overflow: hidden;
}

.feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, transparent, rgba(0, 102, 204, 0.05));
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.feature-card:hover {
  transform: translateY(-5px);
  border-color: var(--border-red);
  box-shadow: 0 8px 24px rgba(0, 102, 204, 0.1);
}

.feature-card:hover::before {
  opacity: 1;
}

.feature-icon {
  font-size: 28px;
  filter: drop-shadow(0 2px 8px rgba(0, 102, 204, 0.5));
  animation: float 3s ease-in-out infinite;
}

.feature-text {
  color: var(--transneft-darker);
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.start-button {
  display: inline-flex;
  align-items: center;
  gap: 15px;
  padding: 18px 40px;
  background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
  color: var(--transneft-white);
  text-decoration: none;
  border-radius: 50px;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.5px;
  transition: all var(--transition-normal);
  box-shadow: 
    0 8px 24px rgba(227, 30, 36, 0.4),
    0 0 0 0 rgba(227, 30, 36, 0.5);
  animation: fadeInUp 1s ease-out 1s both;
  position: relative;
  overflow: hidden;
}

.start-button::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  transform: translate(-50%, -50%);
  transition: width 0.6s, height 0.6s;
}

.start-button:hover::before {
  width: 300px;
  height: 300px;
}

.start-button:hover {
  transform: translateY(-3px) scale(1.05);
  box-shadow: 
    0 12px 40px rgba(227, 30, 36, 0.6),
    0 0 0 4px rgba(227, 30, 36, 0.2);
}

.button-text {
  position: relative;
  z-index: 1;
}

.button-arrow {
  position: relative;
  z-index: 1;
  font-size: 24px;
  transition: transform var(--transition-normal);
}

.start-button:hover .button-arrow {
  transform: translateX(5px);
}

.powered-by {
  margin-top: 40px;
  color: rgba(0, 102, 204, 0.7);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 2px;
  font-weight: 600;
  animation: fadeInUp 1s ease-out 1.2s both;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 768px) {
  .hero-title {
    font-size: 36px;
  }

  .company-name {
    font-size: 32px;
  }

  .hero-subtitle {
    font-size: 16px;
  }

  .logo-circle {
    width: 100px;
    height: 100px;
  }

  .logo-icon {
    width: 100px;
    height: 100px;
  }

  .logo-circle::after {
    width: 16px;
    height: 16px;
  }

  .features-grid {
    flex-direction: column;
    align-items: center;
  }

  .feature-card {
    width: 100%;
    max-width: 300px;
  }

  .start-button {
    padding: 16px 32px;
    font-size: 16px;
  }
}

@media (max-width: 480px) {
  .hero-title {
    font-size: 28px;
  }

  .company-name {
    font-size: 26px;
  }

  .hero-subtitle {
    font-size: 14px;
    margin-bottom: 40px;
  }

  .logo-circle {
    width: 80px;
    height: 80px;
    border-radius: 20px;
  }

  .logo-icon {
    width: 80px;
    height: 80px;
  }

  .logo-circle::after {
    width: 12px;
    height: 12px;
    top: -3px;
    right: -3px;
  }

  .features-grid {
    gap: 15px;
    margin-bottom: 40px;
  }

  .feature-card {
    padding: 16px 24px;
  }

  .feature-icon {
    font-size: 24px;
  }

  .feature-text {
    font-size: 14px;
  }

  .start-button {
    padding: 14px 28px;
    font-size: 15px;
    gap: 10px;
  }

  .button-arrow {
    font-size: 20px;
  }

  .powered-by {
    font-size: 10px;
    margin-top: 30px;
  }
}
</style>