<script setup>
import { RouterLink, RouterView } from 'vue-router';
import { ref } from 'vue';

const isMenuOpen = ref(false);

function toggleMenu() {
  isMenuOpen.value = !isMenuOpen.value;
}
</script>

<template>
  <div class="main-layout">
    <!-- Background -->
    <div class="app-background">
      <div class="grid-overlay"></div>
      <div class="gradient-orb orb-1"></div>
      <div class="gradient-orb orb-2"></div>
    </div>

    <header class="navbar">
      <div class="navbar-container">
        <RouterLink to="/" class="logo-link">
          <div class="logo-section">
            <div class="logo-icon">
              <img src="./assets/logo.svg" alt="Логотип Транснефть" class="logo-svg" />
            </div>
            <div class="logo-text">
              <span class="company-name">Транснефть</span>
              <span class="app-name">AI Консультант</span>
            </div>
          </div>
        </RouterLink>

        <!-- Desktop Navigation -->
        <nav class="desktop-nav">
          <RouterLink to="/" class="nav-link">
            <span class="nav-text">Главная</span>
          </RouterLink>
          <RouterLink to="/chat" class="nav-link">
            <span class="nav-text">Чат-Бот</span>
          </RouterLink>
        </nav>

        <!-- Mobile Menu Button -->
        <button class="mobile-menu-btn" @click="toggleMenu">
          <span class="menu-icon" :class="{ open: isMenuOpen }">
            <span></span>
            <span></span>
            <span></span>
          </span>
        </button>
      </div>

      <!-- Mobile Navigation -->
      <nav class="mobile-nav" :class="{ open: isMenuOpen }">
        <RouterLink to="/" class="nav-link" @click="toggleMenu">
          <span class="nav-icon">🏠</span>
          <span class="nav-text">Главная</span>
        </RouterLink>
        <RouterLink to="/chat" class="nav-link" @click="toggleMenu">
          <span class="nav-icon">💬</span>
          <span class="nav-text">Чат-Бот</span>
        </RouterLink>
      </nav>
    </header>

    <main class="content-area">
      <RouterView />
    </main>

    <footer class="app-footer">
      <div class="footer-content">
        <div class="footer-section">
          <h4>ПАО «Транснефть»</h4>
          <p>Цифровой AI консультант</p>
        </div>
        <div class="footer-section">
          <p class="copyright">© 2025 Credo</p>
        </div>
      </div>
    </footer>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  margin: 0;
  background: linear-gradient(135deg, #001529 0%, #002B5C 50%, #004B93 100%);
  color: #FFFFFF;
  overflow-x: hidden;
}

.main-layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
}

.app-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

.grid-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image: 
    linear-gradient(rgba(0, 75, 147, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 75, 147, 0.05) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
  0% { transform: translate(0, 0); }
  100% { transform: translate(50px, 50px); }
}

.gradient-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(100px);
  opacity: 0.1;
  animation: orbFloat 15s infinite ease-in-out;
}

.orb-1 {
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, #004B93, transparent);
  top: -200px;
  left: -200px;
}

@keyframes orbFloat {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  50% {
    transform: translate(30px, -30px) scale(1.1);
  }
}

.navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: linear-gradient(135deg, rgba(0, 27, 56, 0.95), rgba(0, 43, 92, 0.9));
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(0, 102, 204, 0.3);
  box-shadow: 
    0 4px 30px rgba(0, 0, 0, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.navbar-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 16px 40px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo-link {
  text-decoration: none;
  color: inherit;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 15px;
}

.logo-icon {
  width: 70px;
  height: 70px;
  background: #FFFFFF;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
  box-shadow: 
    0 4px 20px rgba(0, 75, 147, 0.5),
    0 0 0 2px rgba(227, 30, 36, 0.2);
  animation: logoFloat 3s ease-in-out infinite;
  position: relative;
}

.logo-svg {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

@keyframes logoFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.logo-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.company-name {
  font-size: 20px;
  font-weight: 700;
  color: #FFFFFF;
  letter-spacing: -0.5px;
}

.app-name {
  font-size: 12px;
  color: rgba(0, 102, 204, 0.9);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}

.desktop-nav {
  display: flex;
  gap: 10px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  color: rgba(255, 255, 255, 0.8);
  text-decoration: none;
  border-radius: 12px;
  transition: all 0.3s ease;
  font-size: 15px;
  font-weight: 500;
  position: relative;
  overflow: hidden;
}

.nav-link::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 0;
  height: 100%;
  background: linear-gradient(90deg, rgba(0, 102, 204, 0.2), rgba(227, 30, 36, 0.1));
  transition: width 0.3s ease;
}

.nav-link:hover::before {
  width: 100%;
}

.nav-link:hover {
  color: #FFFFFF;
  background: rgba(0, 75, 147, 0.3);
  transform: translateY(-2px);
}

.nav-link.router-link-active {
  color: #FFFFFF;
  background: linear-gradient(135deg, rgba(0, 75, 147, 0.5), rgba(0, 102, 204, 0.4));
  border: 1px solid rgba(227, 30, 36, 0.5);
  box-shadow: 0 0 20px rgba(0, 102, 204, 0.3);
}

.nav-icon {
  font-size: 18px;
  position: relative;
  z-index: 1;
}

.nav-text {
  position: relative;
  z-index: 1;
}

.mobile-menu-btn {
  display: none;
  width: 40px;
  height: 40px;
  background: rgba(0, 75, 147, 0.3);
  border: 1px solid rgba(0, 102, 204, 0.3);
  border-radius: 10px;
  cursor: pointer;
  padding: 0;
  justify-content: center;
  align-items: center;
  transition: all 0.3s ease;
}

.mobile-menu-btn:hover {
  background: rgba(0, 75, 147, 0.5);
  border-color: rgba(227, 30, 36, 0.5);
}

.menu-icon {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.menu-icon span {
  width: 20px;
  height: 2px;
  background: #FFFFFF;
  transition: all 0.3s ease;
  border-radius: 2px;
}

.menu-icon.open span:nth-child(1) {
  transform: rotate(45deg) translateY(7px);
}

.menu-icon.open span:nth-child(2) {
  opacity: 0;
}

.menu-icon.open span:nth-child(3) {
  transform: rotate(-45deg) translateY(-7px);
}

.mobile-nav {
  display: none;
  flex-direction: column;
  gap: 5px;
  padding: 0 20px 20px;
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.mobile-nav.open {
  max-height: 200px;
}

.mobile-nav .nav-link {
  width: 100%;
  justify-content: flex-start;
}

.content-area {
  flex: 1;
  position: relative;
  z-index: 1;
}

.app-footer {
  position: relative;
  z-index: 1;
  background: linear-gradient(135deg, rgba(0, 27, 56, 0.95), rgba(0, 43, 92, 0.9));
  backdrop-filter: blur(20px);
  border-top: 1px solid rgba(0, 102, 204, 0.3);
  padding: 30px 20px;
  margin-top: auto;
}

.footer-content {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 20px;
}

.footer-section h4 {
  font-size: 16px;
  font-weight: 700;
  color: #FFFFFF;
  margin-bottom: 8px;
}

.footer-section p {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.5;
}

.copyright {
  font-size: 12px;
  color: rgba(0, 102, 204, 0.7);
}

@media (max-width: 768px) {
  .navbar-container {
    padding: 16px 20px;
  }

  .desktop-nav {
    display: none;
  }

  .mobile-menu-btn {
    display: flex;
  }

  .mobile-nav {
    display: flex;
  }

  .logo-text {
    display: none;
  }

  .footer-content {
    flex-direction: column;
    text-align: center;
  }
}

@media (max-width: 480px) {
  .logo-icon {
    width: 40px;
    height: 40px;
    font-size: 20px;
  }

  .company-name {
    font-size: 16px;
  }
}
</style>