/* ==========================================================================
   Spark Realtime — Shared State, System State Machine, Sessions & Markdown
   ========================================================================== */

// Configure ONNX Runtime Web to load WASM files locally from /static/vendor/
if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
  ort.env.wasm.wasmPaths = '/static/vendor/';
}

// Global state variables (accessible across modules)
let mediaRecorder = null;
let isRecording = false;
let voiceWs = null;
let audioContext = null;
let activeAudioSources = [];
let masterGainNode = null;
let ttsAborted = false;
let nextPlayTime = null;
let currentTransientMsg = null;
let currentUserMsg = null;
let isManualDisconnect = false;
let pendingSystemPrompt = null;

// Energy gate & VAD settings
let energyThreshold = 0.008;
let vadSensitivityValue = 0.5;
let voiceCallActive = false;
let vadInstance = null;
let voiceCallMuted = false;
let isCurrentlySpeaking = false;
let lastVadSendTime = 0;
const VAD_DEBOUNCE_MS = 300;

// Video call state
let videoCallActive = false;
let videoCallVadInstance = null;
let videoCallMuted = false;
let videoCallCameraOn = true;
let videoCallStream = null;
let videoCallSpeaking = false;
let videoCallProcessing = false;
let lastVideoCallSendTime = 0;
let lastTtsEndTime = 0;
const VIDEO_CALL_DEBOUNCE_MS = 500;
const POST_TTS_COOLDOWN_MS = 250;
let bargeInEnabled = false;
let isTtsPlaying = false;

// Push-to-Talk state (default: hands-free VAD; user override persisted in localStorage)
let pttMode = false;
let pendingHandsFreeSetup = false;
let pendingSessionGreeting = false;
const INPUT_MODE_STORAGE_KEY = 'spark_input_mode';
let pttRecording = false;
let pttProcessing = false;
let pttMediaRecorder = null;
let pttAudioChunks = [];
let pttAudioContext = null;
let pttStream = null;

// Sessions / Chat history state
let currentChatId = null;
let chats = {};
let selectedChatMode = 'chat';
let selectedTemplate = null;
let visionTemplates = {};

// Hermes Agent state
let hermesStatus = {
  configured: false,
  available: false,
  transport: 'unconfigured',
  active_count: 0
};
let hermesTasks = {};
let comfyJobs = {};
let workspaceArtifacts = [];
let activeAgentTab = 'activity';

// System state machine: 'idle' | 'listening' | 'thinking' | 'speaking' | 'agent_working' | 'error'
let currentSystemState = 'idle';
let lastRttMs = null;

function setSystemState(state, customLabel = null) {
  currentSystemState = state;
  const pill = document.getElementById('systemStatePill');
  const labelEl = document.getElementById('systemStateLabel');
  if (!pill || !labelEl) return;

  pill.setAttribute('data-state', state);
  const defaultLabels = {
    idle: 'Idle',
    listening: 'Listening',
    thinking: 'Thinking',
    speaking: 'Speaking',
    agent_working: 'Agent Working',
    error: 'Error'
  };
  labelEl.textContent = customLabel || defaultLabels[state] || 'Idle';

  // Sync mic button visual highlight when listening
  const pttBtn = document.getElementById('pushToTalkBtn');
  if (pttBtn) {
    pttBtn.classList.toggle('listening-active', state === 'listening');
  }
}

function updateLatencyBadge(rttMs) {
  lastRttMs = rttMs;
  const el = document.getElementById('latencyValue');
  if (el) {
    el.textContent = rttMs !== null ? `${Math.round(rttMs)} ms` : '-- ms';
  }
}

function log(msg) {
  const timestamp = new Date().toLocaleTimeString();
  console.log(`[${timestamp}]`, msg);
  const logEl = document.getElementById('log');
  if (logEl) {
    logEl.textContent += `[${timestamp}] ${msg}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function toggleLog() {
  const logEl = document.getElementById('log');
  if (logEl) logEl.classList.toggle('visible');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

// Theme Management
function initTheme() {
  const saved = localStorage.getItem('spark_theme') || 'dark';
  applyTheme(saved);
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('spark_theme', theme);
  const iconContainer = document.getElementById('themeToggleIcon');
  if (iconContainer && window.SparkIcons) {
    iconContainer.innerHTML = window.SparkIcons.iconSvg(theme === 'light' ? 'moon' : 'sun', 15);
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

// Markdown Renderer using vendored marked.min.js
function renderMarkdownPreview(markdown, containerId) {
  if (!markdown) return '';
  if (typeof marked !== 'undefined') {
    marked.setOptions({
      gfm: true,
      breaks: true
    });
    return marked.parse(markdown);
  }
  return `<pre>${escapeHtml(markdown)}</pre>`;
}

function getActiveConversationEl() {
  return document.getElementById('conversation');
}

function removeEmptyState() {
  const activeEl = getActiveConversationEl();
  if (!activeEl) return;
  const emptyState = activeEl.querySelector('.empty-state');
  if (emptyState) emptyState.remove();
}

function scrollToBottom() {
  const activeEl = getActiveConversationEl();
  if (!activeEl) return;
  const doScroll = () => {
    activeEl.scrollTop = activeEl.scrollHeight;
  };
  doScroll();
  requestAnimationFrame(doScroll);
}

let thinkingIndicatorEl = null;

function showThinkingIndicator(label = 'Thinking...') {
  setSystemState('thinking', label.replace('...', ''));
  if (thinkingIndicatorEl) return;
  const activeEl = getActiveConversationEl();
  if (!activeEl) return;

  thinkingIndicatorEl = document.createElement('div');
  thinkingIndicatorEl.className = 'thinking-indicator';
  thinkingIndicatorEl.innerHTML = `
    <div class="thinking-dots">
      <span></span><span></span><span></span>
    </div>
    <span class="thinking-text">${escapeHtml(label)}</span>
  `;
  activeEl.appendChild(thinkingIndicatorEl);
  scrollToBottom();
}

function hideThinkingIndicator() {
  if (thinkingIndicatorEl) {
    thinkingIndicatorEl.remove();
    thinkingIndicatorEl = null;
  }
}

function createMessageElement(role, content = '', isTransient = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message message-${isTransient ? 'transient' : role}`;

  const header = document.createElement('div');
  header.className = 'message-header';
  const roleLabel = isTransient
    ? 'Spark (streaming)'
    : role === 'user'
      ? 'You'
      : role === 'system'
        ? 'System'
        : 'Spark';
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  header.innerHTML = `<span>${escapeHtml(roleLabel)}</span><span>${timeStr}</span>`;

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;

  if (role !== 'system') {
    messageDiv.appendChild(header);
  }
  messageDiv.appendChild(contentDiv);

  return { container: messageDiv, content: contentDiv };
}

function addMessage(role, content) {
  removeEmptyState();
  const activeEl = getActiveConversationEl();
  if (!activeEl) return null;
  const msg = createMessageElement(role, content);
  activeEl.appendChild(msg.container);
  scrollToBottom();
  return msg.container;
}
