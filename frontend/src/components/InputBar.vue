<script setup>
import { ref, computed, defineEmits, defineProps } from 'vue';
import axios from 'axios';

// PROPS & EMITS
const emit = defineEmits(['messageSent', 'setBotStatus']);

const props = defineProps({
});

// ОСТОЯНИЕ (Общее)
const userInput = ref('');
const isLoading = ref(false);
const isRecording = ref(false);
const statusMessage = ref('');

// СОСТОЯНИЕ STT
let mediaRecorder = null;
let audioChunks = [];

// КОНФИГУРАЦИЯ API
const API_BASE_URL = 'http://localhost:8000/api/'; 
const QA_ENDPOINT = 'qa/answer/';
const STT_ENDPOINT = 'stt/recognize/';

const placeholderText = computed(() => {
    if (isLoading.value) return 'Обработка запроса...';
    if (isRecording.value) return '🎙️ Говорите...';
    return 'Введите ваш вопрос о Транснефть...';
});

// QA ФУНКЦИОНАЛ
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isLoading.value) return;
    
    isLoading.value = true;
    statusMessage.value = 'Отправка запроса...';
    emit('setBotStatus', 'thinking');
    
    emit('messageSent', { text, sender: 'user', type: 'text', id: Date.now() });
    userInput.value = '';
    try {
        const response = await axios.post(`${API_BASE_URL}${QA_ENDPOINT}`, { question: text });
        const botAnswer = response.data.answer || 'Извините, ответ не найден.'; 

        statusMessage.value = '';
        emit('messageSent', { 
            text: botAnswer, 
            sender: 'bot', 
            type: 'text',
            id: Date.now()
        });
        
        // Задержка перед сменой статуса на idle
        setTimeout(() => {
            emit('setBotStatus', 'idle');
        }, 5000); // 5 секунд задержки
        
    } catch (error) {
        console.error('Ошибка при запросе к QA API:', error);
        statusMessage.value = '';
        emit('messageSent', { 
            text: 'Извините, произошла ошибка связи с сервером.', 
            sender: 'bot', 
            type: 'error',
            id: Date.now()
        });
        
        // Задержка перед сменой статуса на idle даже при ошибке
        setTimeout(() => {
            emit('setBotStatus', 'idle');
        }, 3000); // 3 секунды при ошибке
        
    } finally {
        isLoading.value = false;
    }
}

// STT ФУНКЦИОНАЛ
async function toggleRecording() {
  if (isRecording.value) {
    mediaRecorder.stop();
    isRecording.value = false;
    statusMessage.value = 'Обработка аудио...';
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());
        await processAudio();
      };
      mediaRecorder.start();
      isRecording.value = true;
      statusMessage.value = 'Запись голоса...';
    } catch (e) {
      console.error('Ошибка доступа к микрофону:', e);
      statusMessage.value = 'Ошибка доступа к микрофону';
      setTimeout(() => statusMessage.value = '', 3000);
    }
  }
}

async function processAudio() {
    if (audioChunks.length === 0) {
        statusMessage.value = '';
        return;
    }

    isLoading.value = true;

    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'question.wav');

    try {
        statusMessage.value = 'Распознавание речи...';
        const response = await axios.post(`${API_BASE_URL}${STT_ENDPOINT}`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        const recognizedText = response.data.text;
        userInput.value = recognizedText;
        statusMessage.value = 'Речь распознана!';
        setTimeout(() => {
          statusMessage.value = '';
          sendMessage();
        }, 500);
    } catch (error) {
        console.error('Ошибка при распознавании речи:', error);
        statusMessage.value = 'Ошибка распознавания речи';
        emit('messageSent', { 
          text: 'Извините, не удалось распознать речь.', 
          sender: 'bot', 
          type: 'error',
          id: Date.now()
        });
        setTimeout(() => statusMessage.value = '', 3000);
    } finally {
        isLoading.value = false;
        audioChunks = [];
    }
}
</script>

<template>
  <div class="input-bar-container">
    <div class="input-wrapper">
      <button 
        @click="toggleRecording" 
        :class="['action-btn', 'mic-btn', { recording: isRecording, disabled: isLoading }]"
        :disabled="isLoading"
        :title="isRecording ? 'Остановить запись' : 'Начать запись (STT)'"
      >
        <svg v-if="!isRecording" class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
        </svg>
        <svg v-else class="btn-icon recording-icon" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="12" cy="12" r="8" />
        </svg>
        <span class="recording-pulse" v-if="isRecording"></span>
      </button>
      
      <div class="input-field-wrapper">
        <input 
          v-model="userInput"
          @keyup.enter="sendMessage"
          :placeholder="placeholderText"
          :disabled="isLoading || isRecording"
          class="input-field"
        />
        <div class="input-border"></div>
      </div>
      
      <button 
        @click="sendMessage" 
        :disabled="!userInput.trim() || isLoading || isRecording"
        :class="['action-btn', 'send-btn', { loading: isLoading }]"
        title="Отправить сообщение"
      >
        <svg v-if="!isLoading" class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
        </svg>
        <span class="loader" v-else></span>
      </button>

      </div>

    <div class="status-bar animate-fadeInUp" v-if="statusMessage">
      <div class="status-indicator"></div>
      <span class="status-text">{{ statusMessage }}</span>
    </div>
  </div>
</template>

<style scoped>
.input-bar-container {
  padding: 20px 24px;
  background: rgba(0, 40, 80, 0.8);
  backdrop-filter: blur(15px);
  border-top: 1px solid rgba(0, 102, 204, 0.2);
}

.input-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}

.input-field-wrapper {
  flex: 1;
  position: relative;
}

.input-field {
  width: 100%;
  padding: 14px 20px;
  background: var(--bg-glass);
  border: 1px solid var(--border-blue);
  border-radius: var(--radius-md);
  color: var(--transneft-white);
  font-size: 15px;
  font-family: inherit;
  transition: all var(--transition-normal);
}

.input-field::placeholder {
  color: rgba(255, 255, 255, 0.5);
}

.input-field:focus {
  outline: none;
  border-color: var(--border-blue-hover);
  background: rgba(0, 75, 147, 0.3);
  box-shadow: 0 0 20px rgba(0, 102, 204, 0.2);
}

.input-field:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.input-border {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--transneft-light-blue), var(--transneft-red));
  transition: width var(--transition-normal);
  border-radius: 2px;
}

.input-field:focus + .input-border {
  width: 100%;
}

.action-btn {
  width: 48px;
  height: 48px;
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  flex-shrink: 0;
  color: var(--transneft-white);
}

.action-btn::before {
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

.action-btn:hover:not(:disabled)::before {
  width: 100px;
  height: 100px;
}

.action-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.action-btn:disabled:hover {
  transform: none;
}

.btn-icon {
  width: 22px;
  height: 22px;
  position: relative;
  z-index: 1;
  transition: all var(--transition-normal);
}

.action-btn:hover:not(:disabled) .btn-icon {
  transform: scale(1.1);
}

.mic-btn {
  background: linear-gradient(135deg, var(--transneft-light-blue), var(--transneft-blue));
  border: 1px solid var(--border-blue);
  box-shadow: var(--shadow-md);
}

.mic-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: var(--shadow-blue);
}

.mic-btn.recording {
  background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
  border-color: var(--border-red);
  animation: recordingPulse 1.5s ease-in-out infinite;
}

@keyframes recordingPulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(227, 30, 36, 0.7);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(227, 30, 36, 0);
  }
}

.recording-icon {
  animation: pulse 1.5s ease-in-out infinite;
}

.recording-pulse {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: var(--radius-md);
  border: 2px solid var(--transneft-red);
  animation: pulseRing 1.5s ease-out infinite;
}

@keyframes pulseRing {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  100% {
    transform: scale(1.4);
    opacity: 0;
  }
}

.send-btn {
  background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
  border: 1px solid var(--border-red);
  box-shadow: var(--shadow-md);
}

.send-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: var(--shadow-red);
}

.send-btn:not(:disabled):active {
  transform: translateY(0) scale(0.95);
}

.send-btn.loading {
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0%, 100% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
}

.speaker-btn {
  background: linear-gradient(135deg, var(--transneft-blue), var(--transneft-medium));
  border: 1px solid var(--border-blue);
  box-shadow: var(--shadow-md);
}

.speaker-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: var(--shadow-blue);
}

.speaker-btn:hover:not(:disabled) .btn-icon {
  animation: soundWave 0.6s ease-in-out;
}

@keyframes soundWave {
  0%, 100% { transform: scale(1); }
  25% { transform: scale(1.1); }
  50% { transform: scale(0.9); }
  75% { transform: scale(1.1); }
}

.loader {
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-top-color: var(--transneft-white);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  position: relative;
  z-index: 1;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
  padding: 10px 16px;
  background: var(--bg-glass);
  border: 1px solid var(--border-blue);
  border-radius: var(--radius-md);
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.status-indicator {
  width: 8px;
  height: 8px;
  background: var(--transneft-light-blue);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--transneft-light-blue);
  animation: statusBlink 1.5s ease-in-out infinite;
}

@keyframes statusBlink {
  0%, 100% { 
    opacity: 1;
    transform: scale(1);
  }
  50% { 
    opacity: 0.3;
    transform: scale(0.8);
  }
}

.status-text {
  color: rgba(255, 255, 255, 0.9);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.3px;
}

@media (max-width: 768px) {
  .input-bar-container {
    padding: 16px;
  }

  .input-wrapper {
    gap: 8px;
  }

  .action-btn {
    width: 44px;
    height: 44px;
  }

  .btn-icon {
    width: 20px;
    height: 20px;
  }

  .input-field {
    padding: 12px 16px;
    font-size: 14px;
  }

  .status-bar {
    padding: 8px 14px;
  }

  .status-text {
    font-size: 12px;
  }
}

@media (max-width: 480px) {
  .input-wrapper {
    gap: 6px;
  }

  .action-btn {
    width: 40px;
    height: 40px;
  }

  .btn-icon {
    width: 18px;
    height: 18px;
  }

  .input-field {
    padding: 10px 14px;
    font-size: 13px;
  }
}
</style>