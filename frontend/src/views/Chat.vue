<template>
  <div class="chat-page-layout">
    <!-- Background -->
    <div class="cyber-background">
      <div class="grid-lines"></div>
      <div class="floating-particles">
        <div class="particle" v-for="i in 20" :key="i" :style="getParticleStyle(i)"></div>
      </div>
      <div class="light-beam beam-1"></div>
      <div class="light-beam beam-2"></div>
      <div class="gradient-orb orb-1"></div>
      <div class="gradient-orb orb-2"></div>
    </div>

    <!-- Main content -->
    <div class="content-wrapper">
      <!-- Left sidebar with avatar -->
      <div class="sidebar-panel glass-effect animate-slideInLeft">
        <div class="avatar-section">
          <div class="avatar-frame">
            <AvatarDisplay :status="botStatus" />
          </div>
        </div>
      </div>

      <!-- Main chat area -->
      <div class="main-panel animate-slideInRight">
        <div class="chat-container glass-effect">
          <button 
            v-if="botStatus !== 'farewell'" 
            @click="handleEndSession" 
            class="end-session-button"
            title="Завершить сеанс"
          >
            <svg class="end-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/>
            </svg>
          </button>

          <div class="chat-topbar">
            <div class="topbar-left">
              <h2 class="chat-title">Трубач</h2>
              <div class="online-indicator">
                <span class="pulse-dot"></span>
                <span class="online-text">Онлайн</span>
              </div>
            </div>
            <div class="topbar-actions">
            </div>
          </div>

          <ChatWindow 
            :messages="messages" 
            @speak-message="handleSpeakRequest" 
          />

          <InputBar 
            @message-sent="handleNewMessage" 
            @set-bot-status="handleSetBotStatus"
            :last-answer="getLastBotAnswerText()"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import AvatarDisplay from '@/components/AvatarDisplay.vue';
import ChatWindow from '@/components/ChatWindow.vue';
import InputBar from '@/components/InputBar.vue';

const router = useRouter();

const messages = ref([
  { id: 0, text: 'Приветствую! Я цифровой консультант Трубач. Чем я могу помочь?', sender: 'bot' }
]);
const nextId = ref(1);
const botStatus = ref('greeting');
const hasGreeted = ref(false);

onMounted(() => {
  setTimeout(() => {
    if (!hasGreeted.value) {
      botStatus.value = 'idle';
      hasGreeted.value = true;
    }
  }, 2500);
});

function getStatusText() {
  const statusMap = {
    greeting: 'Приветствую',
    idle: 'Готов к работе',
    thinking: 'Обработка...',
    farewell: 'До свидания!'
  };
  return statusMap[botStatus.value] || 'В сети';
}

function handleSetBotStatus(status) {
    if (status === 'greeting' && hasGreeted.value) {
      return;
    }
    
    botStatus.value = status;
}

function handleNewMessage(message) {
  messages.value.push({ ...message, id: nextId.value++ });
  
  if (message.sender === 'user') {
    handleSetBotStatus('thinking');
  } else if (message.sender === 'bot') {
    setTimeout(() => {
      handleSetBotStatus('idle');
    }, 2000);
  }
}

function getLastBotAnswerText() {
    const lastBotMsg = messages.value.slice().reverse().find(msg => msg.sender === 'bot');
    return lastBotMsg ? lastBotMsg.text : '';
}

function handleSpeakRequest(text) {
    console.log(`Запуск озвучки текста: "${text}"`);
}

function getParticleStyle(index) {
  return {
    left: `${Math.random() * 100}%`,
    top: `${Math.random() * 100}%`,
    animationDelay: `${Math.random() * 5}s`,
    animationDuration: `${5 + Math.random() * 10}s`
  };
}

async function handleEndSession() {
  botStatus.value = 'farewell';
  
  await new Promise(resolve => setTimeout(resolve, 4000));
  
  router.push('/');
}
</script>

<style scoped>
.chat-page-layout {
  position: relative;
  min-height: 100vh;
  background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 50%, #e3f2fd 100%);
  overflow: hidden;
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
  opacity: 0.3;
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
    transform: translateY(-100vh) translateX(50px);
    opacity: 0;
  }
}

.light-beam {
  position: absolute;
  width: 2px;
  height: 100%;
  background: linear-gradient(180deg, transparent, rgba(0, 102, 204, 0.2), transparent);
  opacity: 0.15;
  animation: beamMove 8s infinite ease-in-out;
}

.beam-1 {
  left: 20%;
  animation-delay: 0s;
}

.beam-2 {
  right: 30%;
  animation-delay: 4s;
}

@keyframes beamMove {
  0%, 100% { transform: translateY(-100%); }
  50% { transform: translateY(100%); }
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
  background: radial-gradient(circle, rgba(0, 102, 204, 0.3), transparent);
  top: -200px;
  left: -200px;
}

.orb-2 {
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(227, 30, 36, 0.2), transparent);
  bottom: -150px;
  right: -150px;
  animation-delay: -7s;
}

@keyframes orbFloat {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  50% {
    transform: translate(30px, -30px) scale(1.1);
  }
}

.content-wrapper {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  height: 100vh;
  padding: 5vh 20px 20px 20px;
  gap: 25px;
}

.sidebar-panel {
  width: 520px;
  height: 95%;
  max-height: 900px;
  border-radius: var(--radius-2xl);
  padding: 35px 25px;
  display: flex;
  flex-direction: column;
  gap: 30px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(0, 102, 204, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
}

.avatar-section {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

.avatar-frame {
  flex-grow: 1;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.avatar-glow {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80%;
  padding-bottom: 80%;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0, 102, 204, 0.2) 0%, transparent 70%);
  transition: all 0.5s ease;
  z-index: 0;
}

.avatar-glow.thinking {
  background: radial-gradient(circle, rgba(227, 30, 36, 0.2) 0%, transparent 70%);
  animation: pulse 1.5s infinite;
}

.status-panel {
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(0, 102, 204, 0.15);
  border-radius: 20px;
  z-index: 2;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #ccc;
  transition: background-color 0.3s ease;
}
.status-dot.idle { background-color: #4CAF50; }
.status-dot.greeting { background-color: #2196F3; animation: pulse 2s infinite; }
.status-dot.thinking { background-color: #f44336; animation: pulse 1s infinite; }
.status-dot.speaking { background-color: #FFC107; }

.status-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--transneft-darker);
}

.info-cards {
  flex-shrink: 0;
}

.main-panel {
  flex: 1;
  max-width: 900px;
  height: 95%;
  max-height: 900px;
  display: flex;
  flex-direction: column;
}

.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  border-radius: var(--radius-2xl);
  overflow: hidden;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(0, 102, 204, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  position: relative;
}

.chat-topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 28px;
  background: linear-gradient(90deg, rgba(255, 255, 255, 0.95), rgba(0, 102, 204, 0.03));
  border-bottom: 1px solid rgba(0, 102, 204, 0.1);
  backdrop-filter: blur(10px);
}

.topbar-left {
  display: flex;
  align-items: center;
  gap: 18px;
}

.chat-title {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: var(--transneft-darker);
  letter-spacing: -0.5px;
}

.online-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  background: rgba(76, 175, 80, 0.1);
  border: 1px solid rgba(76, 175, 80, 0.3);
  border-radius: 18px;
  transition: all var(--transition-normal);
}

.online-indicator:hover {
  background: rgba(76, 175, 80, 0.15);
  box-shadow: 0 0 15px rgba(76, 175, 80, 0.2);
}

.pulse-dot {
  width: 7px;
  height: 7px;
  background: #4CAF50;
  border-radius: 50%;
  box-shadow: 0 0 10px rgba(76, 175, 80, 0.6);
  animation: pulse 2s ease-in-out infinite;
}

.online-text {
  font-size: 11px;
  color: #4CAF50;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.topbar-actions {
  display: flex;
  gap: 8px;
  min-width: 80px;
}

.action-btn {
  width: 36px;
  height: 36px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 102, 204, 0.1);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-normal);
  color: var(--transneft-blue);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
}

.action-btn svg {
  width: 20px;
  height: 20px;
}

.action-btn:hover {
  background: rgba(0, 102, 204, 0.05);
  border-color: rgba(227, 30, 36, 0.3);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15);
}

.end-session-button {
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  background: rgba(227, 30, 36, 0.15);
  border: 1px solid rgba(227, 30, 36, 0.3);
  border-radius: 50%;
  color: rgba(227, 30, 36, 0.7);
  cursor: pointer;
  transition: all var(--transition-normal);
  z-index: 100;
  opacity: 0.6;
}

.end-session-button:hover {
  opacity: 1;
  background: rgba(227, 30, 36, 0.25);
  border-color: rgba(227, 30, 36, 0.6);
  color: var(--transneft-red);
  transform: scale(1.1);
  box-shadow: 0 0 15px rgba(227, 30, 36, 0.4);
}

.end-icon {
  width: 18px;
  height: 18px;
}

@media (max-width: 1200px) {
  .content-wrapper {
    flex-direction: column;
    height: auto;
  }
  
  .sidebar-panel {
    width: 100%;
    flex-direction: row;
    flex-wrap: wrap;
    padding: 25px 20px;
  }
  
  .main-panel {
    max-width: 100%;
    height: 600px;
  }
  
  .avatar-frame {
    width: 100%;
    height: 240px;
  }
}

@media (max-width: 768px) {
  .sidebar-panel {
    width: 100%;
  }
  
  .avatar-frame {
    height: 200px;
  }
  
  .brand-name {
    font-size: 20px;
  }
  
  .chat-title {
    font-size: 18px;
  }
  
  .content-wrapper {
    padding: 10px;
    gap: 15px;
  }
}
</style>