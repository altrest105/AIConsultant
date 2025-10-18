<template>
  <div :class="['avatar-display', status]">
    <div class="avatar-model" ref="avatarContainer">
      <!-- 3D модель -->
      <div class="model-loader" v-if="isLoading">
        <div class="loader-spinner">
          <div class="spinner-ring"></div>
          <div class="spinner-ring"></div>
          <div class="spinner-ring"></div>
        </div>
        <p class="loader-text">Загрузка модели...</p>
        <div class="loader-progress">
          <div class="progress-bar" :style="{ width: loadProgress + '%' }"></div>
        </div>
      </div>
      <div class="model-error" v-if="loadError">
        <svg class="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
        </svg>
        <p class="error-text">Ошибка загрузки модели</p>
        <button @click="retryLoad" class="retry-button">
          <svg class="retry-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          Повторить
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps, defineEmits, computed, ref, onMounted, onUnmounted, watch } from 'vue';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const props = defineProps({
  status: {
    type: String,
    default: 'idle',
  },
});

const emit = defineEmits(['end-session']);

const avatarContainer = ref(null);
const isLoading = ref(true);
const loadError = ref(false);
const loadProgress = ref(0);

let scene, camera, renderer, model, mixer, clock, platform;
let animationFrameId = null;
const animationActions = {};
let activeAction = null;

const statusText = computed(() => {
  switch (props.status) {
    case 'greeting':
      return 'Приветствую!';
    case 'thinking':
      return 'Обработка...';
    case 'speaking':
      return 'Предлагаю задать вопрос';
    case 'farewell':
      return 'До свидания!';
    case 'idle':
    default:
      return 'Готов к работе';
  }
});

function initThreeScene() {
  if (!avatarContainer.value) return;

  scene = new THREE.Scene();
  scene.background = null;

  const width = avatarContainer.value.clientWidth;
  const height = avatarContainer.value.clientHeight;
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 5, 22);
  camera.lookAt(0, 3, -3);

  renderer = new THREE.WebGLRenderer({ 
    antialias: true, 
    alpha: true 
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  avatarContainer.value.appendChild(renderer.domElement);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
  directionalLight.position.set(2, 3, 2);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.width = 2048;
  directionalLight.shadow.mapSize.height = 2048;
  scene.add(directionalLight);

  const rimLight = new THREE.DirectionalLight(0x66ccff, 0.6);
  rimLight.position.set(0, 2, -5);
  scene.add(rimLight);

  const platformGeometry = new THREE.CylinderGeometry(2.5, 2.5, 0.1, 64);
  
  const platformMaterial = new THREE.MeshStandardMaterial({
    color: 0x003366,
    emissive: 0x0066CC,
    emissiveIntensity: 0.2,
    roughness: 0.8,
    metalness: 0.1,
    transparent: true,
    opacity: 0.5
  });

  platform = new THREE.Mesh(platformGeometry, platformMaterial);
  platform.position.y = -1.85;
  platform.receiveShadow = true;
  scene.add(platform);

  clock = new THREE.Clock();

  loadModelAndAnimations();

  animate();

  window.addEventListener('resize', onWindowResize);
}

async function loadModelAndAnimations() {
  const loader = new GLTFLoader();
  const animationFiles = {
    greeting: '/models/hihi.glb',      // приветствие
    idle: '/models/waiting.glb',       // ожидание (основное состояние)
    thinking: '/models/attention.glb', // обработка вопроса
    farewell: '/models/byebye.glb',    // прощание
  };

  try {
    const gltf = await loader.loadAsync('/models/waiting.glb', (progress) => {
      if (progress.total > 0) {
        loadProgress.value = Math.round((progress.loaded / progress.total) * 100 / (Object.keys(animationFiles).length + 1));
      }
    });

    model = gltf.scene;
    model.position.set(0, -1.8, 0);
    model.scale.set(1.2, 1.2, 1.2);
    model.traverse((child) => {
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        if (child.material) {
          child.material.metalness = 0.3;
          child.material.roughness = 0.7;
          child.material.envMapIntensity = 1;
        }
      }
    });
    scene.add(model);

    mixer = new THREE.AnimationMixer(model);
    const idleAction = mixer.clipAction(gltf.animations[0]);
    animationActions['idle'] = idleAction;

    const allPromises = Object.entries(animationFiles).map(([name, path], index) =>
      loader.loadAsync(path, (progress) => {
        if (progress.total > 0) {
          const baseProgress = 100 / (Object.keys(animationFiles).length + 1);
          const currentAnimProgress = (progress.loaded / progress.total) * baseProgress;
          loadProgress.value = Math.round(baseProgress + (index * baseProgress) + currentAnimProgress);
        }
      }).then(animGltf => {
        const clip = animGltf.animations[0];
        if (clip) {
          animationActions[name] = mixer.clipAction(clip);
        }
      })
    );

    await Promise.all(allPromises);

    isLoading.value = false;
    loadProgress.value = 100;
    
    setActiveAnimation('greeting');

  } catch (error) {
    console.error('Ошибка загрузки модели или анимаций:', error);
    isLoading.value = false;
    loadError.value = true;
  }
}

function setActiveAnimation(name, loop = THREE.LoopRepeat) {
  if (!mixer || !animationActions[name]) return;

  const previousAction = activeAction;
  activeAction = animationActions[name];

  if (previousAction && previousAction !== activeAction) {
    previousAction.fadeOut(0.2);
  }

  activeAction
    .reset()
    .setLoop(loop)
    .setEffectiveTimeScale(1)
    .setEffectiveWeight(1)
    .fadeIn(0.2)
    .play();
}

function retryLoad() {
  loadError.value = false;
  isLoading.value = true;
  loadProgress.value = 0;
  if (model) scene.remove(model);
  if (platform) scene.remove(platform);
  for (const key in animationActions) {
    delete animationActions[key];
  }
  if(mixer) mixer.stopAllAction();

  loadModelAndAnimations();
}

function animate() {
  animationFrameId = requestAnimationFrame(animate);
  
  const delta = clock.getDelta();
  
  if (mixer) {
    mixer.update(delta);
  }
  
  if (platform) {
    platform.rotation.y += delta * 0.1;
  }
  
  renderer.render(scene, camera);
}

function onWindowResize() {
  if (!avatarContainer.value) return;
  
  const width = avatarContainer.value.clientWidth;
  const height = avatarContainer.value.clientHeight;
  
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function cleanup() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
  
  window.removeEventListener('resize', onWindowResize);
  
  if (renderer) {
    renderer.dispose();
    if (avatarContainer.value && renderer.domElement) {
      avatarContainer.value.removeChild(renderer.domElement);
    }
  }
  
  if (scene) {
    scene.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach(material => material.dispose());
        } else {
          object.material.dispose();
        }
      }
    });
  }
}

onMounted(() => {
  initThreeScene();
});

onUnmounted(() => {
  cleanup();
});

watch(() => props.status, (newStatus) => {
  if (!model || !mixer) return;
  setActiveAnimation(newStatus);
});
</script>

<style scoped>
.avatar-display {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: relative;
}

.avatar-model {
  width: 100%;
  height: 100%;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: visible;
}

.avatar-model::before {
  content: none;
}

.model-loader {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  z-index: 10;
}

.loader-spinner {
  width: 60px;
  height: 60px;
  position: relative;
}

.spinner-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 3px solid transparent;
  border-top-color: var(--transneft-light-blue);
  border-radius: 50%;
  animation: spinRing 1.5s cubic-bezier(0.5, 0, 0.5, 1) infinite;
}

.spinner-ring:nth-child(1) {
  border-top-color: var(--transneft-blue);
  animation-delay: -0.45s;
}

.spinner-ring:nth-child(2) {
  border-top-color: var(--transneft-light-blue);
  animation-delay: -0.3s;
}

.spinner-ring:nth-child(3) {
  border-top-color: var(--transneft-red);
  animation-delay: -0.15s;
}

@keyframes spinRing {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loader-text {
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0.5px;
  animation: pulse 2s ease-in-out infinite;
}

.loader-progress {
  width: 150px;
  height: 4px;
  background: var(--bg-glass);
  border-radius: 2px;
  overflow: hidden;
  border: 1px solid var(--border-blue);
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--transneft-light-blue), var(--transneft-red));
  border-radius: 2px;
  transition: width 0.3s ease-out;
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { background-position: -150px 0; }
  100% { background-position: 150px 0; }
}

.model-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
  z-index: 10;
  width: 100%;
  padding: 0 20px;
  box-sizing: border-box;
}

.error-icon {
  width: 50px;
  height: 50px;
  color: var(--transneft-red);
  filter: drop-shadow(0 4px 12px rgba(227, 30, 36, 0.5));
  animation: shake 0.5s ease-in-out;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-10px); }
  75% { transform: translateX(10px); }
}

.error-text {
  color: rgba(227, 30, 36, 0.9);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0.3px;
  text-align: center;
}

.retry-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background: linear-gradient(135deg, var(--transneft-red), var(--transneft-dark-red));
  border: 1px solid var(--border-red);
  border-radius: var(--radius-md);
  color: var(--transneft-white);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all var(--transition-normal);
  box-shadow: var(--shadow-md);
}

.retry-button:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-red);
}

.retry-icon {
  width: 16px;
  height: 16px;
  animation: rotate 2s linear infinite;
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.status-indicator {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 24px;
  background: var(--bg-glass);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border-blue);
  border-radius: 30px;
  box-shadow: var(--shadow-lg);
  z-index: 10;
  min-width: 200px;
  justify-content: center;
  transition: width 0.3s ease, min-width 0.3s ease;
}

@keyframes statusFloat {
  0%, 100% {
    transform: translateX(-50%) translateY(0);
  }
  50% {
    transform: translateX(-50%) translateY(-5px);
  }
}

.status-icon {
  width: 20px;
  height: 20px;
  color: var(--transneft-light-blue);
  filter: drop-shadow(0 2px 8px rgba(0, 102, 204, 0.5));
  animation: float 2s ease-in-out infinite;
}

.status-icon svg {
  width: 100%;
  height: 100%;
}

.status-text {
  color: var(--transneft-white);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.avatar-display.greeting .status-indicator {
  border-color: var(--border-blue);
  animation: statusFloat 3s ease-in-out infinite, glow 2s ease-in-out infinite;
}

.avatar-display.greeting .status-icon {
  animation: wave 1s ease-in-out infinite;
}

@keyframes wave {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(-15deg); }
  75% { transform: rotate(15deg); }
}

.avatar-display.thinking .status-indicator {
  border-color: var(--border-red);
  animation: statusFloat 3s ease-in-out infinite, pulse 1.5s ease-in-out infinite;
}

.avatar-display.thinking .status-icon {
  color: var(--transneft-red);
  animation: blink 1.5s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.avatar-display.speaking .status-indicator {
  border-color: var(--border-blue);
  animation: statusFloat 3s ease-in-out infinite, soundWave 0.8s ease-in-out infinite;
}

.avatar-display.speaking .status-icon {
  animation: soundBounce 0.6s ease-in-out infinite;
}

@keyframes soundBounce {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.2); }
}

@keyframes soundWave {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(0, 102, 204, 0.7);
  }
  50% {
    box-shadow: 0 0 0 15px rgba(0, 102, 204, 0);
  }
}

.avatar-display.idle .status-indicator {
  border-color: rgba(0, 102, 204, 0.5);
}

.avatar-display.idle .status-icon {
  color: rgba(0, 102, 204, 0.7);
}

.avatar-display.farewell .status-indicator {
  border-color: var(--border-red);
  animation: statusFloat 3s ease-in-out infinite, fadeOut 2s ease-in-out infinite;
}

@keyframes fadeOut {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

@keyframes glow {
  0%, 100% {
    box-shadow: var(--shadow-blue);
  }
  50% {
    box-shadow: 0 0 30px rgba(0, 102, 204, 0.8);
  }
}

@media (max-width: 768px) {
  .status-indicator {
    padding: 10px 20px;
    bottom: 15px;
  }

  .status-icon {
    width: 18px;
    height: 18px;
  }

  .status-text {
    font-size: 12px;
  }

  .loader-spinner {
    width: 50px;
    height: 50px;
  }

  .error-icon {
    width: 40px;
    height: 40px;
  }

  .loader-progress {
    width: 120px;
  }
}

@media (max-width: 480px) {
  .status-indicator {
    padding: 8px 16px;
    gap: 8px;
  }

  .status-icon {
    width: 16px;
    height: 16px;
  }

  .status-text {
    font-size: 11px;
  }

  .loader-text {
    font-size: 12px;
  }

  .retry-button {
    padding: 8px 16px;
    font-size: 12px;
  }
}
</style>