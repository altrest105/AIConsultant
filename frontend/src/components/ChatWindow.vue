<script setup>
import { defineProps, ref, watch, nextTick } from 'vue';

// КОНФИГУРАЦИЯ API
const API_BASE_URL = 'http://localhost:8000/api/'; 
const TTS_ENDPOINT = 'tts/synthesize/';
// СОСТОЯНИЕ TTS (Потоковое)
const audioQueue = ref([]);
const isTtsPlaying = ref(false);
const audioEl = new Audio();
let isTtsStreaming = false;
const currentPlayingMessageId = ref(null);

// PROPS
const props = defineProps({
  messages: {
    type: Array,
    required: true,
  },
});
const chatContainer = ref(null);

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
}

// НОВАЯ ФУНКЦИЯ: Преобразование кастомных тегов в HTML
function formatBotText(rawText) {
    if (!rawText) return '';

    let htmlText = rawText;

    // 1. Замена заголовков H0, H1, H3, H4 на h3, h4, h5, h6
    // H0 -> h3 (Самый главный заголовок)
    htmlText = htmlText.replace(/<H0>(.*?)<\/H0>/gs, '<h3>$1</h3>');
    // H1 -> h4 (Основной раздел)
    htmlText = htmlText.replace(/<H1>(.*?)<\/H1>/gs, '<h4>$1</h4>');
    // H3 -> h5 (Подраздел)
    htmlText = htmlText.replace(/<H3>(.*?)<\/H3>/gs, '<h5>$1</h5>');
    // H4 -> h6 (Детализация)
    htmlText = htmlText.replace(/<H4>(.*?)<\/H4>/gs, '<h6>$1</h6>');

    // 2. Замена элементов списка L на <li>
    htmlText = htmlText.replace(/<L>(.*?)<\/L>/gs, '<li class="formatted-list-item">$1</li>');

    // 3. Оборачивание всех последовательностей <li> в один <ul>
    // Регулярное выражение ищет одну или более последовательных <li> и оборачивает их в <ul class="formatted-list">
    htmlText = htmlText.replace(/(\s*<li.*?<\/li>\s*)+/gs, '<ul class="formatted-list">$&</ul>');
    
    // 4. Удаление лишних </ul><ul...> которые могли появиться в результате предыдущей замены
    htmlText = htmlText.replace(/<\/ul>\s*<ul class="formatted-list">/gs, '');

    return htmlText.trim();
}

watch(
  () => props.messages,
  () => {
    nextTick(() => {
      if (chatContainer.value) {
        chatContainer.value.scrollTo({
          top: chatContainer.value.scrollHeight,
          behavior: 'smooth'
        });
      }
    });
  },
  { deep: true }
);
// TTS ФУНКЦИОНАЛ
function findRIFF(buffer) {
    const riff = [0x52, 0x49, 0x46, 0x44];
    for (let i = 0; i < buffer.length - 3; i++) {
        if (buffer[i] === riff[0] && 
            buffer[i + 1] === riff[1] && 
            buffer[i + 2] === riff[2] && 
            buffer[i + 3] === riff[3]) {
            return i;
        }
    }
    return -1;
}

function playNextAudio() {
    if (audioQueue.value.length === 0 || isTtsPlaying.value) {
        return;
    }

    isTtsPlaying.value = true;
    
    const { blob } = audioQueue.value.shift();
    const url = URL.createObjectURL(blob);
    audioEl.src = url;

    audioEl.onended = () => {
        URL.revokeObjectURL(url);
        isTtsPlaying.value = false;
        if (isTtsStreaming || audioQueue.value.length > 0) {
             playNextAudio();
        } else {
             stopAudioPlayback(true);
        }
    };

    audioEl.onerror = (e) => {
        console.error('Ошибка воспроизведения аудио-чанка:', e);
        URL.revokeObjectURL(url);
        isTtsPlaying.value = false;
        playNextAudio(); 
    };

    audioEl.play().catch(err => {
        console.error('Ошибка начала воспроизведения:', err);
        isTtsPlaying.value = false;
        playNextAudio();
    });
}

function stopAudioPlayback(resetId = true) {
    isTtsStreaming = false;
    if (audioEl) {
         audioEl.pause();
         audioEl.src = ''; 
         audioEl.onended = null;
         audioEl.onerror = null;
    }

    audioQueue.value = [];
    isTtsPlaying.value = false;
    if (resetId) {
        currentPlayingMessageId.value = null;
    }
}

async function toggleTtsPlayback(msgId, textToSpeak) {
    if (!textToSpeak) return;
    if (isTtsStreaming || isTtsPlaying.value) {
        if (currentPlayingMessageId.value === msgId) {
            stopAudioPlayback(true);
            return;
        } else {
            stopAudioPlayback(false);
        }
    }
    
    currentPlayingMessageId.value = msgId; 
    
    audioQueue.value = [];
    isTtsPlaying.value = false;
    isTtsStreaming = true; 

    try {
        const response = await fetch(`${API_BASE_URL}${TTS_ENDPOINT}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textToSpeak }),
        });
        if (!response.ok) {
            throw new Error(`Ошибка HTTP: ${response.status}`);
        }

        if (!response.body) {
            throw new Error('Ответ не содержит потока данных.');
        }

        const reader = response.body.getReader();
        let buffer = new Uint8Array();
        let chunkIndex = 0;
        
        while (isTtsStreaming) {
            const { done, value } = await reader.read();
            if (done) break;

            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;
            while (true) {
                const riffIndex = findRIFF(buffer);
                if (riffIndex === -1) break; 
                if (buffer.length < riffIndex + 8) break;
                const fileSize = new DataView(buffer.buffer, buffer.byteOffset + riffIndex + 4, 4).getUint32(0, true) + 8;
                if (buffer.length < riffIndex + fileSize) break; 

                const wavData = buffer.slice(riffIndex, riffIndex + fileSize);
                const blob = new Blob([wavData], { type: 'audio/wav' });

                audioQueue.value.push({ blob, index: chunkIndex });
                if (!isTtsPlaying.value) {
                    playNextAudio();
                }
                
                chunkIndex++;
                buffer = buffer.slice(riffIndex + fileSize);
            }
        }
        
    } catch (error) {
        if (isTtsStreaming) { 
            console.error('Ошибка потоковой загрузки/воспроизведения:', error);
        }
    } finally {
        isTtsStreaming = false;
        if (audioQueue.value.length === 0 && !isTtsPlaying.value) {
            stopAudioPlayback(true);
        }
    }
}
</script>

<template>
  <div ref="chatContainer" class="chat-window">
    <div 
      v-for="msg in messages" 
      :key="msg.id" 
      :class="['message-wrapper', msg.sender, 'animate-fadeInUp']"
    >
      <div class="message-container">
        <div class="sender-info">
          <div class="avatar-mini" :class="msg.sender">
            <svg v-if="msg.sender === 'bot'" class="avatar-icon" viewBox="0 0 
24 24" fill="none" stroke="currentColor">
               <rect x="5" y="11" width="14" height="10" rx="2" stroke-width="2"/>
              <circle cx="9" cy="15" r="1" fill="currentColor"/>
              <circle cx="15" cy="15" r="1" fill="currentColor"/>
              <path d="M9 19h6" stroke-linecap="round" stroke-width="2"/>
              <path d="M12 11V8" stroke-linecap="round" stroke-width="2"/>
             <circle cx="12" cy="6" r="2" 
stroke-width="2"/>
            </svg>
            <svg v-else class="avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="8" r="4" stroke-width="2"/>
              <path d="M6 21v-2a4 4 0 0 1 
4-4h4a4 4 0 0 1 4 4v2" stroke-linecap="round" stroke-width="2"/>
            </svg>
          </div>
          <span class="sender-name">{{ msg.sender === 'bot' ?
'Консультант' : 'Вы' }}</span>
          <span class="message-time">{{ getCurrentTime() }}</span>
        </div>
        
        <div class="message-bubble" :class="msg.type || 'text'">
          <div 
             class="bubble-content"
             v-html="msg.sender === 'bot' ? formatBotText(msg.text) : msg.text"
          ></div>
          <div class="bubble-tail"></div>
        </div>
        
        <button 
           v-if="msg.sender === 'bot' && msg.text" 
          @click="toggleTtsPlayback(msg.id, msg.text)" 
          :class="['tts-button', { 'tts-playing': isTtsPlaying && currentPlayingMessageId === msg.id }]"
          :title="isTtsPlaying && currentPlayingMessageId === msg.id ? 'Остановить озвучку' : 'Озвучить сообщение'"
        >
          <svg v-if="isTtsPlaying && currentPlayingMessageId === msg.id" class="tts-icon tts-stop-icon" viewBox="0 0 24 24" fill="currentColor">
             <rect x="6" y="6" width="12" height="12" rx="2" ry="2"></rect>
          </svg>
          <svg v-else class="tts-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 
12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-window {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  background: transparent;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.chat-window::-webkit-scrollbar {
  width: 6px;
}

.chat-window::-webkit-scrollbar-track {
  background: rgba(0, 75, 147, 0.1);
  border-radius: 3px;
}

.chat-window::-webkit-scrollbar-thumb {
  background: var(--border-blue);
  border-radius: 3px;
  transition: background var(--transition-normal);
}

.chat-window::-webkit-scrollbar-thumb:hover {
  background: var(--border-blue-hover);
}

.message-wrapper {
  display: flex;
  animation: messageSlideIn 0.4s ease-out;
}

@keyframes messageSlideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-wrapper.bot {
  justify-content: flex-start;
}

.message-wrapper.user {
  justify-content: flex-end;
}

.message-container {
  display: flex;
  flex-direction: column;
  max-width: 70%;
  position: relative;
}

.sender-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding: 0 12px;
}

.message-wrapper.user .sender-info {
  justify-content: flex-end;
}

.avatar-mini {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  position: relative;
  overflow: hidden;
}

.avatar-mini::before {
  content: '';
  position: absolute;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, transparent, rgba(255, 255, 255, 0.1));
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.avatar-mini:hover::before {
  opacity: 1;
}

.avatar-mini.bot {
  background: linear-gradient(135deg, var(--transneft-blue), var(--transneft-light-blue));
  box-shadow: var(--shadow-blue);
}

.avatar-mini.user {
  background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
  box-shadow: var(--shadow-red);
}

.avatar-icon {
  width: 16px;
  height: 16px;
  color: var(--transneft-white);
  z-index: 1;
  animation: float 4s ease-in-out infinite;
}

.sender-name {
  font-size: 13px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
  letter-spacing: 0.3px;
}

.message-time {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.5);
  margin-left: auto;
}

.message-bubble {
  position: relative;
  padding: 14px 18px;
  border-radius: var(--radius-lg);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
}

.message-bubble:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.message-wrapper.bot .message-bubble {
  background: linear-gradient(135deg, var(--transneft-blue), var(--transneft-light-blue));
  border: 1px solid var(--border-blue);
  border-bottom-left-radius: 4px;
  box-shadow: var(--shadow-blue);
}

.message-wrapper.bot .message-bubble.error {
  background: linear-gradient(135deg, rgba(227, 30, 36, 0.3), rgba(181, 26, 31, 0.2));
  border-color: var(--border-red);
}

.message-wrapper.user .message-bubble {
  background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
  border: 1px solid var(--border-red);
  border-bottom-right-radius: 4px;
}

.bubble-content {
  color: var(--transneft-white);
  font-size: 15px;
  line-height: 1.5;
  word-wrap: break-word;
  position: relative;
  z-index: 1;
}

.bubble-tail {
  position: absolute;
  width: 0;
  height: 0;
  bottom: 0;
}

.message-wrapper.bot .bubble-tail {
  left: -6px;
  border-left: 6px solid transparent;
  border-right: 6px solid rgba(0, 102, 204, 0.3);
  border-bottom: 6px solid transparent;
  border-top: 6px solid rgba(0, 102, 204, 0.3);
}

.message-wrapper.user .bubble-tail {
  right: -6px;
  border-left: 6px solid var(--transneft-dark-red);
  border-right: 6px solid transparent;
  border-bottom: 6px solid transparent;
  border-top: 6px solid var(--transneft-dark-red);
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-2px);
  }
}

.tts-button {
  align-self: flex-start;
  margin-top: 6px;
  margin-left: 8px;
  width: 36px;
  height: 36px;
  background: var(--bg-glass);
  border: 1px solid var(--border-blue);
  border-radius: var(--radius-md);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition-normal);
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;
}

.tts-button::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  transform: translate(-50%, -50%);
  transition: width 0.4s, height 0.4s;
}

.tts-button:hover::before {
  width: 60px;
  height: 60px;
}

.tts-button:hover {
  background: linear-gradient(135deg, rgba(0, 75, 147, 0.5), rgba(0, 102, 204, 0.4));
  border-color: var(--border-red);
  transform: translateY(-2px);
  box-shadow: var(--shadow-blue);
}

.tts-button:active {
  transform: translateY(0) scale(0.95);
}

.tts-icon {
  width: 18px;
  height: 18px;
  color: var(--transneft-white);
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
  position: relative;
  z-index: 1;
  transition: all var(--transition-normal);
}

.tts-button:hover .tts-icon {
  animation: soundWave 0.6s ease-in-out;
}

@keyframes soundWave {
  0%, 100% {
    transform: scale(1);
  }
  25% {
    transform: scale(1.1);
  }
  50% {
    transform: scale(0.9);
  }
  75% {
    transform: scale(1.1);
  }
}

.tts-button.tts-playing {
    background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
    border-color: var(--border-red);
    animation: none;
    transform: none;
}

.tts-button.tts-playing .tts-icon {
    animation: none;
    fill: white;
    stroke: none;
}

.tts-stop-icon {
    width: 16px;
    height: 16px;
    fill: currentColor;
    stroke: none;
}

.bubble-content h3 {
    font-size: 1.2em;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 8px;
    line-height: 1.3;
    color: var(--transneft-white);
}

.bubble-content h4 {
    font-size: 1em;
    font-weight: 600;
    margin-top: 14px;
    margin-bottom: 6px;
    line-height: 1.4;
    color: var(--transneft-white);
}

.bubble-content h5 {
    font-size: 1em;
    font-weight: 500;
    margin-top: 12px;
    margin-bottom: 4px;
    line-height: 1.4;
    color: rgba(255, 255, 255, 0.9);
}

.bubble-content h6 {
    font-size: 0.95em;
    font-weight: 400;
    margin-top: 10px;
    margin-bottom: 2px;
    line-height: 1.4;
    color: rgba(255, 255, 255, 0.7);
}

.bubble-content .formatted-list {
    list-style: none;
    padding-left: 0;
    margin-top: 8px;
    margin-bottom: 8px;
}

.bubble-content .formatted-list-item {
    position: relative;
    padding-left: 18px;
    margin-bottom: 6px;
    line-height: 1.4;
    text-indent: 0;
}

.bubble-content .formatted-list-item::before {
    content: '—'; 
    color: var(--transneft-white);
    position: absolute;
    left: 0;
    font-weight: bold;
    font-size: 1em;
}

.message-bubble.error .bubble-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.message-bubble.error .bubble-content::before {
  content: '';
  display: inline-block;
  width: 18px;
  height: 18px;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23FFFFFF" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>') center/contain no-repeat;
  flex-shrink: 0;
}

.chat-window:empty::before {
  content: 'Начните диалог...';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: rgba(255, 255, 255, 0.4);
  font-size: 18px;
  font-weight: 500;
  pointer-events: none;
}

.typing-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 18px;
  width: 80px;
}

.typing-dot {
  width: 8px;
  height: 8px;
  background: var(--transneft-white);
  border-radius: 50%;
  opacity: 0.7;
  animation: typingDot 1.4s ease-in-out infinite;
}

.typing-dot:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typingDot {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.7;
  }
  30% {
    transform: translateY(-10px);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .chat-window {
    padding: 16px;
    gap: 16px;
  }

  .message-container {
    max-width: 85%;
  }

  .bubble-content {
    font-size: 14px;
  }

  .sender-name {
    font-size: 12px;
  }

  .message-time {
    display: none;
  }

  .avatar-mini {
    width: 24px;
    height: 24px;
  }

  .avatar-icon {
    width: 14px;
    height: 14px;
  }

  .tts-button {
      width: 30px;
      height: 30px;
      margin-top: 2px;
  }

  .tts-icon {
      width: 16px;
      height: 16px;
  }
}
</style>