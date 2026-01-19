// spark-realtime-chatbot Application

// Initialize mermaid for diagram rendering
if (typeof mermaid !== 'undefined') {
  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: { useMaxWidth: true, htmlLabels: true }
  });
}

let mediaRecorder = null;
let isRecording = false;
let voiceWs = null;
let audioContext = null;
let activeAudioSources = []; // Track active audio sources for barge-in
let masterGainNode = null; // For instant muting on barge-in
let ttsAborted = false; // Block new audio after barge-in
let nextPlayTime = null;
let currentTransientMsg = null;
let currentUserMsg = null;  // Track current user message being built

const logEl = document.getElementById("log");
const connectionStatusEl = document.getElementById("connectionStatus");
const conversationContainerEl = document.getElementById("conversationContainer");
const conversationEl = document.getElementById("conversation");
const videoConversationEl = document.getElementById("videoConversation");
const pushToTalkBtn = document.getElementById("pushToTalkBtn");
const clearBtn = document.getElementById("clearBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const voiceSelect = document.getElementById("voiceSelect");
const systemPromptInput = document.getElementById("systemPromptInput");
const savePromptBtn = document.getElementById("savePromptBtn");
const chatListEl = document.getElementById("chatList");
const textInput = document.getElementById("textInput");
const sendTextBtn = document.getElementById("sendTextBtn");

// Mobile sidebar elements
const chatSidebar = document.getElementById("chatSidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");

// Mobile sidebar toggle functions
function toggleMobileSidebar() {
  if (chatSidebar && sidebarOverlay) {
    chatSidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('active');
  }
}

function closeMobileSidebar() {
  if (chatSidebar && sidebarOverlay) {
    chatSidebar.classList.remove('open');
    sidebarOverlay.classList.remove('active');
  }
}

// Close sidebar when selecting a chat on mobile
function onChatItemClick() {
  if (window.innerWidth <= 768) {
    closeMobileSidebar();
  }
}

// ========== Energy Gate for Noise Robustness ==========
// Minimum RMS energy threshold to accept VAD speech detection
// Values: 0.001 = very sensitive, 0.01 = moderate, 0.02 = strict
let energyThreshold = 0.008;

/**
 * Calculate RMS (Root Mean Square) energy of audio samples.
 * @param {Float32Array} audioSamples - Audio samples in range [-1, 1]
 * @returns {number} RMS energy value
 */
function calculateRmsEnergy(audioSamples) {
  if (!audioSamples || audioSamples.length === 0) return 0;

  let sumSquares = 0;
  for (let i = 0; i < audioSamples.length; i++) {
    sumSquares += audioSamples[i] * audioSamples[i];
  }
  return Math.sqrt(sumSquares / audioSamples.length);
}

/**
 * Check if audio has enough energy to be considered speech.
 * @param {Float32Array} audioSamples - Audio samples
 * @param {number} threshold - Energy threshold (default: energyThreshold)
 * @returns {boolean} True if energy is above threshold
 */
function hasEnoughEnergy(audioSamples, threshold = energyThreshold) {
  const energy = calculateRmsEnergy(audioSamples);
  const passed = energy >= threshold;
  if (!passed) {
    log(`Energy gate: Rejected (RMS=${energy.toFixed(4)}, threshold=${threshold})`);
  } else {
    log(`Energy gate: Passed (RMS=${energy.toFixed(4)})`);
  }
  return passed;
}

/**
 * Update energy threshold from UI slider.
 * @param {number|string} value - New threshold value
 */
function updateEnergyThreshold(value) {
  energyThreshold = parseFloat(value);
  // Update all display elements
  const displays = ['energyThresholdValue', 'videoEnergyThresholdValue'];
  displays.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = energyThreshold.toFixed(3);
  });
  // Sync both sliders
  const sliders = ['energyThreshold', 'videoEnergyThreshold'];
  sliders.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = energyThreshold;
  });
  log(`Energy threshold updated to ${energyThreshold.toFixed(3)}`);
}

/**
 * Update energy meter visualization.
 * @param {number} energy - Current RMS energy value
 * @param {string} mode - 'voice' or 'video'
 */
function updateEnergyMeter(energy, mode = 'voice') {
  const barId = mode === 'video' ? 'videoEnergyBar' : 'voiceEnergyBar';
  const valueId = mode === 'video' ? 'videoEnergyValue' : 'voiceEnergyValue';
  const bar = document.getElementById(barId);
  const valueEl = document.getElementById(valueId);

  if (bar) {
    // Scale energy to percentage (0.05 = 100%)
    const percent = Math.min(100, (energy / 0.05) * 100);
    bar.style.width = `${percent}%`;
    // Color based on threshold
    if (energy < energyThreshold) {
      bar.style.background = '#e17055'; // Red - below threshold
    } else {
      bar.style.background = '#00b894'; // Green - above threshold
    }
  }
  if (valueEl) {
    valueEl.textContent = energy.toFixed(3);
  }
}

// Helper to get the active conversation element based on current mode
function getActiveConversationEl() {
  // Check if video call is active
  const wrapper = document.getElementById('videoChatWrapper');
  if (wrapper && wrapper.classList.contains('active')) {
    return videoConversationEl || conversationEl;
  }
  return conversationEl;
}

// Chat management state
let currentChatId = null;  // Current active chat ID
let chats = {};  // Store all chats: { chatId: { id, title, preview, timestamp, messages: [] } }

// Chat management functions
function generateChatId() {
  return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function saveChatsToStorage() {
  try {
    localStorage.setItem('spark_realtime_chats', JSON.stringify(chats));
  } catch (e) {
    console.error('Failed to save chats to localStorage:', e);
  }
}

function clearPreviousChats() {
  // Get all chat IDs except the current one
  const chatIds = Object.keys(chats).filter(id => id !== currentChatId);

  if (chatIds.length === 0) {
    log('No previous chats to clear');
    return;
  }

  if (!confirm(`Delete ${chatIds.length} previous chat${chatIds.length > 1 ? 's' : ''}? This cannot be undone.`)) {
    return;
  }

  // Delete all chats except the current one
  chatIds.forEach(id => delete chats[id]);
  saveChatsToStorage();
  renderChatList();
  log(`Cleared ${chatIds.length} previous chat(s)`);
}

function deleteChat(chatId, event) {
  // Prevent the click from selecting the chat
  if (event) {
    event.stopPropagation();
  }

  // Confirm deletion
  if (!confirm('Delete this chat? This cannot be undone.')) {
    return;
  }
  
  // Delete from chats object
  if (chats[chatId]) {
    delete chats[chatId];
    saveChatsToStorage();
    
    // If we deleted the current chat, switch to another or create new
    if (currentChatId === chatId) {
      const remainingChats = Object.keys(chats);
      if (remainingChats.length > 0) {
        // Load the most recent remaining chat
        const sortedChats = remainingChats.sort((a, b) => 
          (chats[b].timestamp || 0) - (chats[a].timestamp || 0)
        );
        loadChat(sortedChats[0]);
      } else {
        // No chats left, create a new one
        currentChatId = null;
        openChatModeModal();
      }
    }
    
    renderChatList();
    log(`Chat deleted: ${chatId}`);
  }
}

function loadChatsFromStorage() {
  try {
    const stored = localStorage.getItem('spark_realtime_chats');
    if (stored) {
      chats = JSON.parse(stored);
      renderChatList();
    }
  } catch (e) {
    console.error('Failed to load chats from localStorage:', e);
  }
}

// =============================================
// Chat Mode Selection
// =============================================

let selectedChatMode = 'call';
let selectedTemplate = null;

function openChatModeModal() {
  log("Opening chat mode modal");
  const modal = document.getElementById('chatModeModal');
  if (!modal) {
    log("ERROR: chatModeModal not found!");
    return;
  }
  modal.classList.add('active');
  modal.style.display = 'flex';  // Force display in case CSS isn't applying
  selectedChatMode = 'call';
  selectedTemplate = null;
  updateModeSelection();
  log("Modal opened");
}

function closeChatModeModal() {
  const modal = document.getElementById('chatModeModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';  // Force hide
  }
}

function selectChatMode(mode) {
  console.log('📋 [SelectMode] selectChatMode called with mode:', mode);
  selectedChatMode = mode;
  selectedTemplate = null;
  updateModeSelection();
}

function updateModeSelection() {
  document.querySelectorAll('.chat-mode-card').forEach(card => {
    card.classList.toggle('selected', card.dataset.mode === selectedChatMode);
  });
}

function startSelectedChat() {
  console.log('🚀 [StartChat] startSelectedChat() called, selectedChatMode:', selectedChatMode);
  closeChatModeModal();
  closeMobileSidebar();

  // Save current chat if exists
  if (currentChatId && chats[currentChatId]) {
    saveCurrentChat();
  }

  // Teardown any existing modes
  teardownVoiceCallMode();
  teardownVideoCallMode();

  // Create new chat with mode info
  const chatId = generateChatId();
  const now = new Date();

  // Determine chat title based on mode
  let chatTitle = 'New Chat';
  if (selectedChatMode === 'call') {
    chatTitle = '📞 Voice Call';
  } else if (selectedChatMode === 'video') {
    chatTitle = '📹 Video Call';
  }

  chats[chatId] = {
    id: chatId,
    title: chatTitle,
    preview: '',
    timestamp: now.toISOString(),
    messages: [],
    mode: selectedChatMode
  };

  currentChatId = chatId;
  saveChatsToStorage();
  renderChatList();
  loadChat(currentChatId);

  // Setup mode-specific UI
  if (selectedChatMode === 'call') {
    setupVoiceCallMode();
  } else if (selectedChatMode === 'video') {
    console.log('🎥 [StartChat] Video mode selected, calling setupVideoCallMode()');
    setupVideoCallMode();
  }

  // Clear conversation UI with appropriate message
  if (selectedChatMode === 'call') {
    conversationEl.innerHTML = `<div class="empty-state">
      <span class="call-mode-badge">📞 Voice Call</span>
      <br><br>Just start talking - Spark will listen and respond automatically.
    </div>`;
  } else if (selectedChatMode === 'video') {
    conversationEl.innerHTML = `<div class="empty-state">
      <span class="video-call-mode-badge">📹 Video Call</span>
      <br><br>Show and ask - Spark sees what you show and hears what you say.
    </div>`;
  } else {
    conversationEl.innerHTML = '<div class="empty-state">Connect and start a conversation</div>';
  }
  currentTransientMsg = null;

  log(`New ${selectedChatMode} chat created: ${chatId}`);
}

// ========================================
// Voice Call Mode with VAD
// ========================================

let voiceCallActive = false;
let vadInstance = null;
let voiceCallMuted = false;
let voiceCallAudioContext = null;
let voiceCallAnalyser = null;
let waveformBars = [];
let audioChunks = [];
let isCurrentlySpeaking = false;
let lastVadSendTime = 0;
const VAD_DEBOUNCE_MS = 300; // Prevent duplicate sends within 300ms

async function setupVoiceCallMode() {
  log('Setting up Voice Call mode with VAD...');
  
  const container = document.getElementById('voiceCallContainer');
  const textInputContainer = document.getElementById('textInputContainer');
  
  // Show voice call UI, hide text input
  container.classList.add('active');
  if (textInputContainer) textInputContainer.style.display = 'none';
  
  // Initialize waveform bars
  initWaveformBars();
  
  try {
    // Check if VAD library is available
    if (typeof vad === 'undefined') {
      throw new Error('VAD library not loaded. Please refresh the page.');
    }
    
    // Initialize VAD
    vadInstance = await vad.MicVAD.new({
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      redemptionFrames: 8,
      preSpeechPadFrames: 10,
      minSpeechFrames: 3,
      
      onSpeechStart: () => {
        if (voiceCallMuted) return;
        log('VAD: Speech started');

        // BARGE-IN: If TTS is playing and barge-in is enabled, stop it
        if (isTtsPlaying && bargeInEnabled) {
          log('VAD: Barge-in triggered - stopping TTS');
          stopTtsPlayback();
          updateVoiceCallStatus('hearing', 'Interrupted - listening...');
        } else {
          updateVoiceCallStatus('hearing', 'Hearing you...');
        }

        // Reset current user message for new speech input
        currentUserMsg = null;

        isCurrentlySpeaking = true;
        audioChunks = [];
      },
      
      onSpeechEnd: (audio) => {
        if (voiceCallMuted) return;

        // Calculate and display energy level
        const energy = calculateRmsEnergy(audio);
        updateEnergyMeter(energy, 'voice');

        // Debounce to prevent duplicate sends
        const now = Date.now();
        if (now - lastVadSendTime < VAD_DEBOUNCE_MS) {
          log('VAD: Debounced duplicate speech end');
          return;
        }
        lastVadSendTime = now;

        // Energy gate - reject low-energy audio (noise)
        if (energy < energyThreshold) {
          log(`VAD: Rejected by energy gate (RMS=${energy.toFixed(4)}, threshold=${energyThreshold})`);
          isCurrentlySpeaking = false;
          updateVoiceCallStatus('listening', 'Listening...');
          return;
        }
        log(`VAD: Energy gate passed (RMS=${energy.toFixed(4)})`);

        log('VAD: Speech ended, processing audio...');
        isCurrentlySpeaking = false;
        updateVoiceCallStatus('processing', 'Processing...');

        // Convert Float32Array to WAV and send
        sendVoiceCallAudio(audio);
      },
      
      onVADMisfire: () => {
        log('VAD: Misfire (too short)');
        if (!isCurrentlySpeaking) {
          updateVoiceCallStatus('listening', 'Listening...');
        }
      }
    });
    
    // Start VAD
    vadInstance.start();
    voiceCallActive = true;
    updateVoiceCallStatus('listening', 'Listening...');
    log('Voice Call mode active with VAD');
    
    // Start waveform animation
    startWaveformAnimation();
    
  } catch (error) {
    log(`VAD initialization error: ${error.message}`);
    console.error('VAD error:', error);
    
    // Fallback message
    updateVoiceCallStatus('listening', 'VAD unavailable - using fallback');
    
    // Show error in UI
    addMessage('system', `⚠️ Voice Activity Detection could not initialize: ${error.message}. Falling back to push-to-talk mode.`);
  }
}

function initWaveformBars() {
  const waveform = document.getElementById('voiceWaveform');
  waveform.innerHTML = '';
  waveformBars = [];
  
  const numBars = 20;
  for (let i = 0; i < numBars; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = '10px';
    waveform.appendChild(bar);
    waveformBars.push(bar);
  }
}

function startWaveformAnimation() {
  if (!voiceCallActive) return;
  
  // Animate bars based on voice activity
  const animate = () => {
    if (!voiceCallActive) return;
    
    waveformBars.forEach((bar, i) => {
      let height;
      if (isCurrentlySpeaking) {
        // Active speaking - random heights
        height = Math.random() * 60 + 20;
      } else {
        // Idle - gentle wave
        const time = Date.now() / 1000;
        height = Math.sin(time * 2 + i * 0.3) * 10 + 15;
      }
      bar.style.height = `${height}px`;
    });
    
    requestAnimationFrame(animate);
  };
  
  animate();
}

function updateVoiceCallStatus(state, text) {
  const statusEl = document.getElementById('voiceCallStatus');
  const textEl = document.getElementById('voiceCallStatusText');
  
  statusEl.className = `voice-call-status ${state}`;
  textEl.textContent = text;
}

async function sendVoiceCallAudio(audioFloat32) {
  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    log('WebSocket not connected, cannot send audio');
    updateVoiceCallStatus('listening', 'Not connected');
    return;
  }
  
  try {
    // Convert Float32Array to WAV blob
    const wavBlob = float32ToWav(audioFloat32, 16000);
    
    // Convert to base64 or send as binary
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = reader.result.split(',')[1];
      
      // Send audio for ASR
      voiceWs.send(JSON.stringify({
        type: 'asr_audio',
        audio: base64,
        format: 'wav'
      }));
      
      log('Sent voice call audio for ASR');
    };
    reader.readAsDataURL(wavBlob);
    
  } catch (error) {
    log(`Error sending voice call audio: ${error.message}`);
    updateVoiceCallStatus('listening', 'Error - Listening...');
  }
}

function float32ToWav(float32Array, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32Array.length * 2);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + float32Array.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, float32Array.length * 2, true);
  
  // Convert float32 to int16
  let offset = 44;
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

function toggleVoiceCallMute() {
  voiceCallMuted = !voiceCallMuted;
  const btn = document.getElementById('voiceCallMuteBtn');
  
  if (voiceCallMuted) {
    btn.classList.add('muted');
    btn.textContent = '🔇';
    updateVoiceCallStatus('listening', 'Muted');
  } else {
    btn.classList.remove('muted');
    btn.textContent = '🎤';
    updateVoiceCallStatus('listening', 'Listening...');
  }
  
  log(`Voice call muted: ${voiceCallMuted}`);
}

function updateVadSensitivity(value) {
  log(`VAD sensitivity updated to: ${value}`);
  // Note: VAD sensitivity would need to be re-initialized to change
  // For now, just log the value
}

function endVoiceCall() {
  log('Ending voice call mode');
  teardownVoiceCallMode();

  // Hide voice call container
  const container = document.getElementById('voiceCallContainer');
  container.classList.remove('active');

  // Show conversation container without text input
  const conversationContainer = document.getElementById('conversationContainer');
  const textInputContainer = document.getElementById('textInputContainer');
  if (conversationContainer) conversationContainer.style.display = 'flex';
  if (textInputContainer) textInputContainer.style.display = 'none';

  addMessage('system', '📞 Voice call ended. Click "+ New" to start another call.');
}

function teardownVoiceCallMode() {
  voiceCallActive = false;

  // Stop any playing TTS
  stopTtsPlayback();

  if (vadInstance) {
    try {
      vadInstance.pause();
      vadInstance.destroy();
    } catch (e) {
      console.error('Error destroying VAD:', e);
    }
    vadInstance = null;
  }
  
  if (voiceCallAudioContext) {
    voiceCallAudioContext.close();
    voiceCallAudioContext = null;
  }
  
  const container = document.getElementById('voiceCallContainer');
  if (container) container.classList.remove('active');
  
  log('Voice call mode torn down');
}

// ========================================
// Video Call Mode (VAD + Webcam + VLM)
// ========================================

let videoCallActive = false;
let videoCallVadInstance = null;
let videoCallMuted = false;
let videoCallCameraOn = true;
let videoCallStream = null;
let videoCallWaveformBars = [];
let videoCallSpeaking = false;
let videoCallProcessing = false; // Flag to block VAD while processing a request
let lastVideoCallSendTime = 0;
let lastTtsEndTime = 0; // Track when TTS finished
const VIDEO_CALL_DEBOUNCE_MS = 500; // 0.5 second debounce to prevent duplicate sends
const POST_TTS_COOLDOWN_MS = 250; // 0.25 second cooldown after TTS ends before accepting new speech

// Barge-in support
let bargeInEnabled = false; // Default off to avoid accidental interrupts
let isTtsPlaying = false;

// Push-to-Talk support
let pttMode = false; // false = VAD mode, true = PTT mode
let pttRecording = false;
let pttProcessing = false; // Flag to prevent duplicate processing
let pttMediaRecorder = null;
let pttAudioChunks = [];
let pttAudioContext = null;
let pttStream = null;

// Default video call system prompt
const DEFAULT_VIDEO_CALL_PROMPT = `You are on a live video call. You can see the user. Respond ONLY to what they ask.

RULES:
- Answer ONLY the specific question asked
- Do NOT describe the scene unless asked
- Do NOT mention things the user didn't ask about
- Keep responses brief and natural (spoken aloud via TTS)
- If user says "okay", "thanks", "got it" - just acknowledge briefly

Be a helpful friend on a video call, not a surveillance camera.`;

async function setupVideoCallMode() {
  log('Setting up Video Call mode with VAD + Webcam...');
  
  const wrapper = document.getElementById('videoChatWrapper');
  const container = document.getElementById('videoCallContainer');
  const textInputContainer = document.getElementById('textInputContainer');
  const regularConversation = document.getElementById('conversationContainer');
  const mainContent = document.querySelector('.main-content');
  
  // Show video call wrapper (side-by-side layout)
  if (wrapper) wrapper.classList.add('active');
  if (container) container.classList.add('active');
  if (mainContent) mainContent.classList.add('video-call-expanded');
  if (textInputContainer) textInputContainer.style.display = 'none';
  if (regularConversation) regularConversation.style.display = 'none';
  
  // Initialize waveform bars
  initVideoCallWaveformBars();
  
  // Start webcam
  try {
    const video = document.getElementById('videoCallWebcam');
    videoCallStream = await navigator.mediaDevices.getUserMedia({ 
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false // Audio handled by VAD
    });
    video.srcObject = videoCallStream;
    await video.play();
    log('Video call webcam started');
  } catch (err) {
    log(`Webcam error: ${err.message}`);
    document.getElementById('videoCallWebcamStatus').textContent = '❌ Camera Error';
  }
  
  // Initialize VAD
  try {
    if (typeof vad === 'undefined') {
      throw new Error('VAD library not loaded');
    }
    
    videoCallVadInstance = await vad.MicVAD.new({
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      redemptionFrames: 8,
      preSpeechPadFrames: 10,
      minSpeechFrames: 3,
      
      onSpeechStart: () => {
        if (videoCallMuted) return;
        
        // Ignore VAD in PTT mode (safety check)
        if (pttMode) {
          log('Video VAD: Ignoring - PTT mode active');
          return;
        }
        
        log(`Video VAD: Speech started (isTtsPlaying: ${isTtsPlaying}, bargeInEnabled: ${bargeInEnabled}, processing: ${videoCallProcessing})`);
        
        // Don't update UI if we're going to ignore this speech anyway
        if (videoCallProcessing || (isTtsPlaying && !bargeInEnabled)) {
          log('Video VAD: Speech will be ignored, not updating UI');
          return;
        }
        
        // BARGE-IN: If TTS is playing and barge-in is enabled, stop it
        if (isTtsPlaying && bargeInEnabled) {
          log('Video VAD: Barge-in triggered - stopping TTS');
          stopTtsPlayback();
          updateVideoCallStatus('hearing', 'Interrupted - listening...');
        } else {
          updateVideoCallStatus('hearing', 'Hearing you...');
        }

        // Reset current user message for new speech input
        currentUserMsg = null;

        videoCallSpeaking = true;
      },
      
      onSpeechEnd: (audio) => {
        if (videoCallMuted) return;

        // Calculate and display energy level
        const energy = calculateRmsEnergy(audio);
        updateEnergyMeter(energy, 'video');

        // Ignore VAD in PTT mode (safety check)
        if (pttMode) {
          log('Video VAD: Ignoring speech end - PTT mode active');
          return;
        }

        // Don't process if already processing a request
        if (videoCallProcessing) {
          log('Video VAD: Ignoring speech - still processing previous request');
          return;
        }

        // Don't process if TTS is playing
        if (isTtsPlaying) {
          log('Video VAD: Ignoring speech end during TTS playback');
          return;
        }

        const now = Date.now();

        // Post-TTS cooldown - wait before accepting new speech after a response
        if (lastTtsEndTime > 0 && now - lastTtsEndTime < POST_TTS_COOLDOWN_MS) {
          log(`Video VAD: Post-TTS cooldown (${POST_TTS_COOLDOWN_MS - (now - lastTtsEndTime)}ms remaining)`);
          return;
        }

        // Debounce to prevent duplicate sends
        if (now - lastVideoCallSendTime < VIDEO_CALL_DEBOUNCE_MS) {
          log('Video VAD: Debounced duplicate speech end');
          return;
        }

        // Energy gate - reject low-energy audio (noise)
        if (energy < energyThreshold) {
          log(`Video VAD: Rejected by energy gate (RMS=${energy.toFixed(4)}, threshold=${energyThreshold})`);
          videoCallSpeaking = false;
          updateVideoCallStatus('listening', 'Listening...');
          return;
        }
        log(`Video VAD: Energy gate passed (RMS=${energy.toFixed(4)})`);

        log('Video VAD: Speech ended, capturing frame + audio...');
        videoCallSpeaking = false;
        videoCallProcessing = true; // Set processing flag
        updateVideoCallStatus('processing', 'Looking & thinking...');

        // Capture frame and send with audio
        sendVideoCallData(audio);
      },
      
      onVADMisfire: () => {
        if (!videoCallSpeaking) {
          updateVideoCallStatus('listening', 'Listening...');
        }
      }
    });
    
    // Only start VAD if not in PTT mode
    if (!pttMode) {
      videoCallVadInstance.start();
      updateVideoCallStatus('listening', 'Listening...');
    } else {
      // In PTT mode, keep VAD paused
      videoCallVadInstance.pause();
      updateVideoCallStatus('listening', 'Press SPACE or hold button to talk');
    }
    videoCallActive = true;
    log('Video Call mode active (PTT mode: ' + pttMode + ')');
    
    startVideoCallWaveformAnimation();
    
  } catch (error) {
    log(`VAD error: ${error.message}`);
    updateVideoCallStatus('listening', 'VAD error');
  }
}

function initVideoCallWaveformBars() {
  const waveform = document.getElementById('videoCallWaveform');
  waveform.innerHTML = '';
  videoCallWaveformBars = [];
  
  for (let i = 0; i < 15; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = '8px';
    waveform.appendChild(bar);
    videoCallWaveformBars.push(bar);
  }
}

function startVideoCallWaveformAnimation() {
  const animate = () => {
    if (!videoCallActive) return;
    
    videoCallWaveformBars.forEach((bar, i) => {
      let height;
      if (videoCallSpeaking) {
        height = Math.random() * 40 + 15;
      } else {
        const time = Date.now() / 1000;
        height = Math.sin(time * 2 + i * 0.3) * 8 + 12;
      }
      bar.style.height = `${height}px`;
    });
    
    requestAnimationFrame(animate);
  };
  animate();
}

function updateVideoCallStatus(state, text) {
  const statusEl = document.getElementById('videoCallStatus');
  const textEl = document.getElementById('videoCallStatusText');
  
  // Reuse voice-call-status classes for styling
  statusEl.className = `video-call-status voice-call-status ${state}`;
  textEl.textContent = text;
}

async function sendVideoCallData(audioFloat32) {
  // Debounce to prevent duplicate sends
  const now = Date.now();
  if (now - lastVideoCallSendTime < VIDEO_CALL_DEBOUNCE_MS) {
    log('Video call: Debounced duplicate send');
    return;
  }
  lastVideoCallSendTime = now;
  
  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    log('WebSocket not connected');
    updateVideoCallStatus('listening', 'Not connected');
    return;
  }
  
  // PAUSE VAD immediately when sending - will resume after TTS completes
  if (videoCallVadInstance && !pttMode) {
    try {
      videoCallVadInstance.pause();
      log('VAD paused while processing');
      
      // Safety fallback: resume VAD after 30 seconds if TTS never comes
      // (e.g., server error, tool call without TTS response)
      setTimeout(() => {
        if (videoCallProcessing && !isTtsPlaying && videoCallVadInstance && videoCallActive && !pttMode) {
          videoCallProcessing = false;
          try {
            videoCallVadInstance.start();
            log('VAD resumed (safety fallback after 30s)');
            updateVideoCallStatus('listening', 'Listening...');
          } catch (e) {}
        }
      }, 30000);
    } catch (e) {}
  }
  
  try {
    // Capture frame from webcam
    const video = document.getElementById('videoCallWebcam');
    let imageBase64 = null;
    
    if (videoCallCameraOn && video.srcObject) {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      imageBase64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      log(`Captured frame: ${canvas.width}x${canvas.height}`);
    }
    
    // Convert audio to WAV
    const wavBlob = float32ToWav(audioFloat32, 16000);
    const reader = new FileReader();
    
    reader.onload = async () => {
      const audioBase64 = reader.result.split(',')[1];
      
      // Send video call data with both audio and image
      const payload = {
        type: 'video_call_data',
        audio: audioBase64,
        image: imageBase64,
        format: 'wav'
      };
      
      // Include system prompt from main config (if set)
      const mainPrompt = document.getElementById('systemPromptInput')?.value?.trim();
      if (mainPrompt) {
        payload.system_prompt = mainPrompt;
      }
      
      voiceWs.send(JSON.stringify(payload));
      log('Sent video call data (audio + frame)');
    };
    reader.readAsDataURL(wavBlob);
    
  } catch (error) {
    log(`Video call send error: ${error.message}`);
    updateVideoCallStatus('listening', 'Error');
  }
}

function toggleVideoCallMute() {
  videoCallMuted = !videoCallMuted;
  const btn = document.getElementById('videoCallMuteBtn');
  
  if (videoCallMuted) {
    btn.classList.add('muted');
    btn.textContent = '🔇';
    updateVideoCallStatus('listening', 'Muted');
  } else {
    btn.classList.remove('muted');
    btn.textContent = '🎤';
    // Show appropriate status based on input mode
    if (pttMode) {
      updateVideoCallStatus('listening', 'Press SPACE or hold button to talk');
    } else {
      updateVideoCallStatus('listening', 'Listening...');
    }
  }
}

function toggleVideoCallCamera() {
  videoCallCameraOn = !videoCallCameraOn;
  const btn = document.getElementById('videoCallCameraBtn');
  const status = document.getElementById('videoCallWebcamStatus');
  const video = document.getElementById('videoCallWebcam');
  
  if (videoCallCameraOn) {
    btn.classList.remove('off');
    btn.textContent = '📷';
    status.textContent = '📹 Camera On';
    if (videoCallStream) {
      videoCallStream.getVideoTracks().forEach(t => t.enabled = true);
    }
  } else {
    btn.classList.add('off');
    btn.textContent = '📷';
    status.textContent = '📷 Camera Off';
    if (videoCallStream) {
      videoCallStream.getVideoTracks().forEach(t => t.enabled = false);
    }
  }
}

function endVideoCall() {
  log('Ending video call mode');
  teardownVideoCallMode();

  const wrapper = document.getElementById('videoChatWrapper');
  const container = document.getElementById('videoCallContainer');
  const textInputContainer = document.getElementById('textInputContainer');
  const regularConversation = document.getElementById('conversationContainer');
  const mainContent = document.querySelector('.main-content');

  // Hide video call wrapper, show regular conversation without text input
  wrapper.classList.remove('active');
  container.classList.remove('active');
  if (mainContent) mainContent.classList.remove('video-call-expanded');
  if (textInputContainer) textInputContainer.style.display = 'none';
  if (regularConversation) regularConversation.style.display = 'flex';

  addMessage('system', '📹 Video call ended. Click "+ New" to start another call.');
}

function teardownVideoCallMode() {
  videoCallActive = false;

  // Stop any playing TTS
  stopTtsPlayback();

  if (videoCallVadInstance) {
    try {
      videoCallVadInstance.pause();
      videoCallVadInstance.destroy();
    } catch (e) {}
    videoCallVadInstance = null;
  }
  
  if (videoCallStream) {
    videoCallStream.getTracks().forEach(t => t.stop());
    videoCallStream = null;
  }
  
  // Cleanup PTT resources
  cleanupPttAudio();
  pttMode = false;
  
  const wrapper = document.getElementById('videoChatWrapper');
  const container = document.getElementById('videoCallContainer');
  const regularConversation = document.getElementById('conversationContainer');
  const mainContent = document.querySelector('.main-content');
  
  if (wrapper) wrapper.classList.remove('active');
  if (container) container.classList.remove('active');
  if (mainContent) mainContent.classList.remove('video-call-expanded');
  if (regularConversation) regularConversation.style.display = '';
  
  const video = document.getElementById('videoCallWebcam');
  if (video) video.srcObject = null;
  
  // Hide settings panel and PTT container
  const settingsPanel = document.getElementById('videoCallSettingsPanel');
  if (settingsPanel) settingsPanel.style.display = 'none';
  
  const pttContainer = document.getElementById('videoCallPttContainer');
  if (pttContainer) pttContainer.style.display = 'none';
  
  // Reset waveform visibility
  const waveform = document.getElementById('videoCallWaveform');
  if (waveform) waveform.style.display = 'flex';
  
  log('Video call mode torn down');
}

function toggleVideoCallSettings() {
  const panel = document.getElementById('videoCallSettingsPanel');
  if (panel.style.display === 'none') {
    panel.style.display = 'block';
  } else {
    panel.style.display = 'none';
  }
}

// ========================================
// Push-to-Talk (PTT) Functions
// ========================================

function setInputMode(mode) {
  const vadBtn = document.getElementById('vadModeBtn');
  const pttBtn = document.getElementById('pttModeBtn');
  const pttContainer = document.getElementById('videoCallPttContainer');
  const waveform = document.getElementById('videoCallWaveform');
  
  if (mode === 'ptt') {
    pttMode = true;
    // Update button styles
    vadBtn.style.background = 'transparent';
    vadBtn.style.color = '#00b894';
    vadBtn.style.fontWeight = 'normal';
    pttBtn.style.background = '#00b894';
    pttBtn.style.color = 'white';
    pttBtn.style.fontWeight = 'bold';
    
    // Show PTT button, hide waveform
    if (pttContainer) pttContainer.style.display = 'block';
    if (waveform) waveform.style.display = 'none';
    
    // Pause VAD when in PTT mode
    if (videoCallVadInstance) {
      try {
        videoCallVadInstance.pause();
        log('VAD paused for PTT mode');
      } catch (e) {}
    }
    
    // Initialize PTT audio stream
    initPttAudio();
    
    updateVideoCallStatus('listening', 'Press SPACE or hold button to talk');
    log('Switched to PTT mode');
  } else {
    pttMode = false;
    // Update button styles
    vadBtn.style.background = '#00b894';
    vadBtn.style.color = 'white';
    vadBtn.style.fontWeight = 'bold';
    pttBtn.style.background = 'transparent';
    pttBtn.style.color = '#00b894';
    pttBtn.style.fontWeight = 'normal';
    
    // Hide PTT button, show waveform
    if (pttContainer) pttContainer.style.display = 'none';
    if (waveform) waveform.style.display = 'flex';
    
    // Resume VAD when in VAD mode
    if (videoCallVadInstance) {
      try {
        videoCallVadInstance.start();
        log('VAD resumed');
      } catch (e) {}
    }
    
    // Cleanup PTT audio
    cleanupPttAudio();
    
    updateVideoCallStatus('listening', 'Listening...');
    log('Switched to VAD mode');
  }
}

async function initPttAudio() {
  try {
    if (pttStream) return; // Already initialized
    
    pttStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true
      }
    });
    
    pttAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    log('PTT audio initialized');
  } catch (err) {
    log(`PTT audio init error: ${err.message}`);
  }
}

function cleanupPttAudio() {
  if (pttMediaRecorder && pttMediaRecorder.state !== 'inactive') {
    pttMediaRecorder.stop();
  }
  pttMediaRecorder = null;
  
  if (pttStream) {
    pttStream.getTracks().forEach(t => t.stop());
    pttStream = null;
  }
  
  if (pttAudioContext) {
    pttAudioContext.close();
    pttAudioContext = null;
  }
  
  pttAudioChunks = [];
  pttRecording = false;
  pttProcessing = false;
}

let pttLastTouchTime = 0; // Track last touch event to ignore subsequent mouse events

async function startPttRecording(event) {
  // Prevent touch events from also triggering mouse events
  if (event) {
    if (event.type === 'touchstart') {
      event.preventDefault();
      pttLastTouchTime = Date.now();
    } else if (event.type === 'mousedown' && Date.now() - pttLastTouchTime < 500) {
      // Ignore mousedown that follows touchstart within 500ms
      return;
    }
  }
  
  if (!pttMode || pttRecording || videoCallMuted) return;
  
  log('PTT: Starting recording');
  pttRecording = true;
  pttAudioChunks = [];
  
  // Visual feedback
  const pttBtn = document.getElementById('videoCallPttBtn');
  if (pttBtn) pttBtn.classList.add('recording');
  updateVideoCallStatus('hearing', 'Recording...');
  
  // BARGE-IN: If TTS is playing and barge-in is enabled, stop it
  if (isTtsPlaying && bargeInEnabled) {
    log('PTT: Barge-in triggered - stopping TTS');
    stopTtsPlayback();
  }
  
  try {
    if (!pttStream) {
      await initPttAudio();
    }
    
    // Create MediaRecorder with WAV-compatible format
    const options = { mimeType: 'audio/webm;codecs=opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      // Fallback
      options.mimeType = 'audio/webm';
    }
    
    pttMediaRecorder = new MediaRecorder(pttStream, options);
    
    pttMediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        pttAudioChunks.push(e.data);
      }
    };
    
    pttMediaRecorder.onstop = async () => {
      // Prevent duplicate processing
      if (pttProcessing) {
        log('PTT: Already processing, ignoring duplicate onstop');
        return;
      }
      
      if (pttAudioChunks.length === 0) {
        log('PTT: No audio recorded');
        return;
      }
      
      pttProcessing = true;
      log('PTT: Processing recorded audio...');
      updateVideoCallStatus('processing', 'Looking & thinking...');
      
      // Convert to blob and then to Float32Array for consistency with VAD
      const audioBlob = new Blob(pttAudioChunks, { type: 'audio/webm' });
      
      // Clear chunks immediately to prevent reprocessing
      pttAudioChunks = [];
      
      try {
        await processPttAudio(audioBlob);
      } finally {
        pttProcessing = false;
      }
    };
    
    pttMediaRecorder.start(100); // Collect data every 100ms
    
  } catch (err) {
    log(`PTT recording error: ${err.message}`);
    pttRecording = false;
    const pttBtn = document.getElementById('videoCallPttBtn');
    if (pttBtn) pttBtn.classList.remove('recording');
    updateVideoCallStatus('listening', 'Recording error');
  }
}

function stopPttRecording(event) {
  // Prevent touch events from also triggering mouse events
  if (event) {
    if (event.type === 'touchend') {
      event.preventDefault();
      pttLastTouchTime = Date.now();
    } else if ((event.type === 'mouseup' || event.type === 'mouseleave') && Date.now() - pttLastTouchTime < 500) {
      // Ignore mouse events that follow touch within 500ms
      return;
    }
  }
  
  if (!pttMode || !pttRecording) return;
  
  log('PTT: Stopping recording');
  pttRecording = false;
  
  // Visual feedback
  const pttBtn = document.getElementById('videoCallPttBtn');
  if (pttBtn) pttBtn.classList.remove('recording');
  
  // Stop the recorder (this triggers onstop which processes the audio)
  if (pttMediaRecorder && pttMediaRecorder.state !== 'inactive') {
    pttMediaRecorder.stop();
  }
}

async function processPttAudio(audioBlob) {
  try {
    // Decode the webm audio to Float32Array
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Get the audio data as Float32Array
    const channelData = audioBuffer.getChannelData(0);
    
    // Resample to 16kHz if needed
    let audioFloat32;
    if (audioBuffer.sampleRate !== 16000) {
      const ratio = 16000 / audioBuffer.sampleRate;
      const newLength = Math.round(channelData.length * ratio);
      audioFloat32 = new Float32Array(newLength);
      
      for (let i = 0; i < newLength; i++) {
        const srcIndex = i / ratio;
        const srcIndexFloor = Math.floor(srcIndex);
        const srcIndexCeil = Math.min(srcIndexFloor + 1, channelData.length - 1);
        const t = srcIndex - srcIndexFloor;
        audioFloat32[i] = channelData[srcIndexFloor] * (1 - t) + channelData[srcIndexCeil] * t;
      }
    } else {
      audioFloat32 = channelData;
    }
    
    await audioContext.close();
    
    // Send using the existing video call data function
    sendVideoCallData(audioFloat32);
    
  } catch (err) {
    log(`PTT audio processing error: ${err.message}`);
    updateVideoCallStatus('listening', 'Press SPACE or hold button to talk');
  }
}

// Keyboard handling for PTT
let pttKeyDown = false;  // Track if PTT key is currently held down

function setupPttKeyboardHandlers() {
  document.addEventListener('keydown', (e) => {
    // Only handle Space key when in PTT mode and video call is active
    // IMPORTANT: Check e.repeat to ignore key repeat events (held key)
    if (e.code === 'Space' && pttMode && videoCallActive && !pttRecording && !e.repeat && !pttKeyDown) {
      // Prevent default space behavior (scrolling, etc)
      e.preventDefault();
      pttKeyDown = true;
      startPttRecording();
    }
  });
  
  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && pttMode && videoCallActive) {
      e.preventDefault();
      pttKeyDown = false;
      if (pttRecording) {
        stopPttRecording();
      }
    }
  });
  
  // Handle window blur - stop recording if user switches away while holding key
  window.addEventListener('blur', () => {
    if (pttKeyDown && pttRecording) {
      log('PTT: Window lost focus, stopping recording');
      pttKeyDown = false;
      stopPttRecording();
    }
  });
  
  log('PTT keyboard handlers initialized');
}

// Initialize PTT keyboard handlers on page load
document.addEventListener('DOMContentLoaded', setupPttKeyboardHandlers);

function stopTtsPlayback() {
  log(`stopTtsPlayback called - active sources: ${activeAudioSources.length}, masterGain: ${masterGainNode ? 'exists' : 'null'}`);
  
  // Set abort flag to block any incoming audio chunks
  ttsAborted = true;
  
  // INSTANT MUTE via gain node
  if (masterGainNode && audioContext) {
    masterGainNode.gain.setValueAtTime(0, audioContext.currentTime);
    log('Master gain set to 0 (instant mute)');
  }
  
  // Stop all active audio sources
  if (activeAudioSources.length > 0) {
    log(`Stopping ${activeAudioSources.length} active audio sources`);
    activeAudioSources.forEach(source => {
      try {
        source.stop();
        source.disconnect();
      } catch (e) {
        // Source may have already ended
      }
    });
    activeAudioSources = [];
  }
  
  // Reset playback timing
  nextPlayTime = null;
  
  // Tell server to abort TTS
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'abort_tts' }));
    log('Sent abort_tts to server');
  }
  
  isTtsPlaying = false;
  log('TTS playback stopped (barge-in complete)');
  
  // Resume VAD after barge-in (immediate since user is already speaking)
  if (videoCallActive && videoCallVadInstance && !pttMode) {
    try {
      videoCallVadInstance.start();
      log('VAD resumed after barge-in');
    } catch (e) {}
  }
}

function addMessage(role, content) {
  // Remove empty state if present
  const emptyState = conversationEl.querySelector('.empty-state');
  if (emptyState) emptyState.remove();
  
  const msgDiv = document.createElement('div');
  msgDiv.className = `message message-${role}`;
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;
  
  msgDiv.appendChild(contentDiv);
  conversationEl.appendChild(msgDiv);
  conversationEl.scrollTop = conversationEl.scrollHeight;
  
  return msgDiv;
}

// Override createNewChat to open modal instead
function createNewChat() {
  log("New Chat button clicked");
  openChatModeModal();
}

function saveCurrentChat() {
  if (!currentChatId) return;

  // Extract messages from the active conversation UI
  // Use videoConversationEl if video call is active, otherwise use conversationEl
  const activeEl = getActiveConversationEl();
  const messages = [];
  const messageElements = activeEl.querySelectorAll('.message');
  
  messageElements.forEach(msgEl => {
    const role = msgEl.classList.contains('message-user') ? 'user' : 'assistant';
    const contentEl = msgEl.querySelector('.message-content');
    if (contentEl) {
      // Check if this is a code message (has code block)
      const codeBlock = contentEl.querySelector('pre');
      if (codeBlock) {
        // Extract task and code
        const taskDiv = contentEl.querySelector('div');
        const task = taskDiv ? taskDiv.textContent.replace('Task: ', '') : '';
        const code = codeBlock.textContent.trim();
        // Store as formatted message
        messages.push({ 
          role, 
          content: `🤖 Coding Assistant\nTask: ${task}\n\n\`\`\`python\n${code}\n\`\`\`` 
        });
      } else {
        const text = contentEl.textContent.trim();
        if (text) {
          messages.push({ role, content: text });
        }
      }
    }
  });
  
  // Update chat
  if (chats[currentChatId]) {
    // Only update timestamp if messages actually changed (new content added)
    const oldMsgCount = chats[currentChatId].messages?.length || 0;
    const hasNewMessages = messages.length > oldMsgCount;

    chats[currentChatId].messages = messages;
    // Update title and preview
    const firstUserMsg = messages.find(m => m.role === 'user');
    if (firstUserMsg) {
      chats[currentChatId].title = firstUserMsg.content.substring(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '');
      chats[currentChatId].preview = firstUserMsg.content.substring(0, 100);
    }
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg.role === 'assistant') {
      chats[currentChatId].preview = lastMsg.content.substring(0, 100);
    }
    // Only update timestamp when new messages are added
    if (hasNewMessages) {
      chats[currentChatId].timestamp = new Date().toISOString();
    }
  }
  
  saveChatsToStorage();
  renderChatList();
}

function renderChatList() {
  if (!chatListEl) return;
  
  // Sort chats by timestamp (newest first)
  const sortedChats = Object.values(chats).sort((a, b) => 
    new Date(b.timestamp) - new Date(a.timestamp)
  );
  
  chatListEl.innerHTML = '';
  
  sortedChats.forEach(chat => {
    const chatItem = document.createElement('div');
    chatItem.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
    chatItem.onclick = () => { loadChat(chat.id); onChatItemClick(); };
    
    const time = new Date(chat.timestamp);
    const timeStr = time.toLocaleDateString() + ' ' + time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    chatItem.innerHTML = `
      <div class="chat-item-content">
        <div class="chat-item-title">${escapeHtml(chat.title || 'New Chat')}</div>
        <div class="chat-item-preview">${escapeHtml(chat.preview || 'No messages yet')}</div>
        <div class="chat-item-time">${timeStr}</div>
      </div>
      <button class="chat-item-delete" onclick="deleteChat('${chat.id}', event)" title="Delete chat">🗑️</button>
    `;
    
    chatListEl.appendChild(chatItem);
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function loadChat(chatId) {
  // Save current chat first
  if (currentChatId && chats[currentChatId] && currentChatId !== chatId) {
    saveCurrentChat();
  }
  
  // Teardown any active modes from previous chat
  teardownVoiceCallMode();
  teardownVideoCallMode();
  
  // Disconnect if connected
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    isManualDisconnect = true;
    voiceWs.send(JSON.stringify({ type: "disconnect" }));
    voiceWs.close();
    voiceWs = null;
  }
  
  currentChatId = chatId;
  renderChatList();
  
  // Clear conversation UI
  conversationEl.innerHTML = '<div class="empty-state">Connect and start a conversation</div>';
  if (videoConversationEl) {
    videoConversationEl.innerHTML = '';
  }
  currentTransientMsg = null;

  // Hide/show text input based on mode
  const textInputContainer = document.getElementById('textInputContainer');

  // Load messages from chat - always render to main conversation container
  // (Don't auto-start call modes for old chats - user can manually restart)
  const chat = chats[chatId];

  if (chat && chat.messages && chat.messages.length > 0) {
    conversationEl.innerHTML = '';
    chat.messages.forEach(msg => {
      const msgEl = createMessageElement(msg.role, msg.content);
      conversationEl.appendChild(msgEl.container);
    });
    scrollToBottom();
  } else if (chat && chat.mode === 'call') {
    // Show voice call mode empty state with restart hint
    conversationEl.innerHTML = `<div class="empty-state">
      <span class="call-mode-badge">📞 Voice Call</span>
      <br><br>This was a voice call chat.<br>Select "Voice Call" mode to start a new call.
    </div>`;
  } else if (chat && chat.mode === 'video') {
    // Show video call mode empty state with restart hint
    conversationEl.innerHTML = `<div class="empty-state">
      <span class="video-call-mode-badge">📹 Video Call</span>
      <br><br>This was a video call chat.<br>Select "Video Call" mode to start a new call.
    </div>`;
  }

  // Show text input for viewing old chats (don't auto-start call modes)
  if (textInputContainer) textInputContainer.style.display = 'flex';
  
  // Update system prompt display based on mode
  updateSystemPromptDisplay(chat);
  
  log(`Loaded chat: ${chatId}${chat?.mode === 'call' ? ' (voice call mode)' : chat?.mode === 'video' ? ' (video call mode)' : ''}`);
}

function updateSystemPromptDisplay(chat) {
  const systemPromptInput = document.getElementById('systemPromptInput');
  const systemPromptMode = document.getElementById('systemPromptMode');

  if (!systemPromptInput) return;

  // Show editable prompt
  systemPromptInput.style.backgroundColor = '';
  // Only enable if disconnected
  const isConnected = voiceWs && voiceWs.readyState === WebSocket.OPEN;
  systemPromptInput.disabled = isConnected;

  if (systemPromptMode) {
    systemPromptMode.textContent = isConnected ? '(Connected - Read Only)' : '(Editable)';
    systemPromptMode.style.color = '#7f8c8d';
  }
}

function scrollToBottom() {
  // Use multiple methods to ensure scroll works
  const doScroll = () => {
    const activeEl = getActiveConversationEl();
    if (activeEl) {
      // Force scroll to bottom
      activeEl.scrollTop = activeEl.scrollHeight;
      
      // Also try scrollIntoView on last child
      const lastChild = activeEl.lastElementChild;
      if (lastChild) {
        lastChild.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }
    
    // Also scroll the regular conversationEl just in case
    if (conversationEl && conversationEl !== activeEl) {
      conversationEl.scrollTop = conversationEl.scrollHeight;
    }
  };
  
  // Call immediately
  doScroll();
  
  // Also call after a short delay to handle any rendering delays
  requestAnimationFrame(doScroll);
  setTimeout(doScroll, 100);
  setTimeout(doScroll, 300);
}

function log(msg) {
  const timestamp = new Date().toLocaleTimeString();
  console.log(`[${timestamp}]`, msg);
  if (logEl) {
    logEl.textContent += `[${timestamp}] ${msg}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function toggleLog() {
  logEl.classList.toggle("visible");
}

function toggleConfig() {
  const configContent = document.getElementById("configContent");
  const configSection = document.querySelector(".config-section");
  const isVisible = configContent.style.display !== "none";
  
  configContent.style.display = isVisible ? "none" : "block";
  configSection.classList.toggle("expanded", !isVisible);
}

function setConnectionStatus(status) {
  connectionStatusEl.className = `connection-status ${status}`;
  const dot = connectionStatusEl.querySelector(".status-dot");
  dot.className = `status-dot ${status}`;
  
  const statusText = {
    connected: "Connected",
    disconnected: "Disconnected",
    connecting: "Connecting..."
  };
  connectionStatusEl.querySelector("span:last-child").textContent = statusText[status];
  
  // Enable/disable system prompt editing and update disconnect button text based on connection status
  if (status === "connected") {
    systemPromptInput.disabled = true;
    disconnectBtn.textContent = "Disconnect";
    disconnectBtn.disabled = false;
    disconnectBtn.classList.remove("connect-state");
  } else if (status === "connecting") {
    // Keep prompt disabled while connecting
    systemPromptInput.disabled = true;
    disconnectBtn.textContent = "Connecting...";
    disconnectBtn.disabled = true;
  } else {
    // Disconnected
    systemPromptInput.disabled = false;
    disconnectBtn.textContent = "Connect";
    disconnectBtn.disabled = false;
    disconnectBtn.classList.add("connect-state");
  }
  
  // Update text input state
  updateTextInputState();
}

function sendTextMessage() {
  if (!textInput || !textInput.value.trim()) {
    return;
  }

  const messageText = textInput.value.trim();

  // Regular text chat mode
  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    alert("Please connect first before sending messages");
    return;
  }
  
  // Clear input
  textInput.value = '';
  textInput.style.height = 'auto';
  
  // Remove empty state
  removeEmptyState();
  
  // Create user message in UI
  const userMsg = createMessageElement("user", messageText);
  conversationEl.appendChild(userMsg.container);
  scrollToBottom();
  
  // Save chat after user message
  saveCurrentChat();
  
  // Send to server
  voiceWs.send(JSON.stringify({
    type: "text_message",
    text: messageText
  }));

  // Show thinking indicator
  showThinkingIndicator();

  log(`Sent text message: "${messageText}"`);
}

function updateTextInputState() {
  if (textInput && sendTextBtn) {
    const isConnected = voiceWs && voiceWs.readyState === WebSocket.OPEN;
    textInput.disabled = !isConnected;
    sendTextBtn.disabled = !isConnected;
  }
}

function removeEmptyState() {
  const activeEl = getActiveConversationEl();
  const emptyState = activeEl.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }
}

// Thinking indicator for perceived responsiveness
let thinkingIndicatorEl = null;

function showThinkingIndicator() {
  // Don't show if already showing
  if (thinkingIndicatorEl) return;

  const activeEl = getActiveConversationEl();

  thinkingIndicatorEl = document.createElement("div");
  thinkingIndicatorEl.className = "thinking-indicator";
  thinkingIndicatorEl.innerHTML = `
    <div class="thinking-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
    <span class="thinking-text">Thinking...</span>
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

function createMessageElement(role, content = "", isTransient = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message message-${isTransient ? 'transient' : role}`;
  
  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = isTransient ? "Assistant (thinking...)" : (role === "user" ? "You" : "Assistant");
  
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = content;
  
  messageDiv.appendChild(header);
  messageDiv.appendChild(contentDiv);
  
  return { container: messageDiv, content: contentDiv };
}

function createMarkdownMessageElement(task, markdown) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message message-assistant";
  
  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = "📝 Markdown Assistant";
  
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.style.padding = "1rem";
  contentDiv.style.backgroundColor = "#f8f9fa";
  contentDiv.style.borderRadius = "8px";
  contentDiv.style.border = "1px solid #e0e0e0";
  
  // Task description
  const taskDiv = document.createElement("div");
  taskDiv.style.marginBottom = "0.75rem";
  taskDiv.style.fontWeight = "500";
  taskDiv.style.color = "#2c3e50";
  taskDiv.textContent = `📄 ${task}`;
  contentDiv.appendChild(taskDiv);
  
  // Rendered markdown preview (truncated)
  const previewDiv = document.createElement("div");
  const previewId = 'md-preview-' + Math.random().toString(36).substr(2, 9);
  previewDiv.id = previewId;
  previewDiv.style.padding = "1rem";
  previewDiv.style.background = "white";
  previewDiv.style.borderRadius = "6px";
  previewDiv.style.border = "1px solid #e9ecef";
  previewDiv.style.maxHeight = "200px";
  previewDiv.style.overflowY = "auto";
  previewDiv.style.lineHeight = "1.5";
  previewDiv.innerHTML = renderMarkdownPreview(markdown.substring(0, 1000) + (markdown.length > 1000 ? '...' : ''), previewId);
  contentDiv.appendChild(previewDiv);
  
  // Character count
  const charCount = document.createElement("div");
  charCount.style.marginTop = "0.5rem";
  charCount.style.fontSize = "0.85rem";
  charCount.style.color = "#666";
  charCount.textContent = `${markdown.length} characters generated`;
  contentDiv.appendChild(charCount);
  
  messageDiv.appendChild(header);
  messageDiv.appendChild(contentDiv);
  
  return { container: messageDiv, content: contentDiv };
}

function connectVoiceWebSocket() {
  console.log(`[connectVoiceWebSocket] Function called`);
  log(`connectVoiceWebSocket called. Current state: voiceWs=${voiceWs ? voiceWs.readyState : 'null'}`);
  
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    log("WebSocket already connected");
    return;
  }
  
  if (voiceWs && (voiceWs.readyState === WebSocket.CONNECTING || voiceWs.readyState === WebSocket.CLOSING)) {
    log("WebSocket already connecting/closing, waiting...");
    return;
  }

  // Close existing connection if any
  if (voiceWs) {
    try {
      voiceWs.close();
      voiceWs = null;
    } catch (e) {
      // Ignore errors closing old connection
    }
  }

  log(`Connecting to WebSocket: ${location.protocol === "https:" ? "wss://" : "ws://"}${location.host}/ws/voice`);
  setConnectionStatus("connecting");
  const wsProtocol = (location.protocol === "https:") ? "wss://" : "ws://";
  voiceWs = new WebSocket(wsProtocol + location.host + "/ws/voice");
  
  // Add connection timeout
  const connectionTimeout = setTimeout(() => {
    if (voiceWs && voiceWs.readyState !== WebSocket.OPEN) {
      log("Connection timeout - server may not be running");
      setConnectionStatus("disconnected");
      try {
        voiceWs.close();
      } catch (e) {
        // Ignore errors
      }
    }
  }, 10000); // 10 second timeout

  voiceWs.onopen = () => {
    clearTimeout(connectionTimeout);
    log("Voice WebSocket connected");
    setConnectionStatus("connected");
    pushToTalkBtn.disabled = false;
    // Send pending system prompt if we have one, otherwise get current one
    setTimeout(() => {
      if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        if (pendingSystemPrompt) {
          // Send the new system prompt
          voiceWs.send(JSON.stringify({ type: "set_system_prompt", prompt: pendingSystemPrompt }));
          log(`System prompt set: ${pendingSystemPrompt.substring(0, 50)}...`);
          pendingSystemPrompt = null;
        } else {
          // Request current system prompt
          voiceWs.send(JSON.stringify({ type: "get_system_prompt" }));
        }
        // Set initial voice from dropdown selection
        const initialVoice = voiceSelect.value;
        voiceWs.send(JSON.stringify({ type: "set_voice", voice: initialVoice }));
        
        // Send initial tool state (capabilities + agents)
        const capabilityCheckboxes = document.querySelectorAll('input[id^="cap"]');
        const agentCheckboxes = document.querySelectorAll('input[id^="agent"]');
        const enabledTools = [];
        
        capabilityCheckboxes.forEach(cb => {
          if (cb.checked) {
            enabledTools.push(cb.value);
          }
        });
        
        agentCheckboxes.forEach(cb => {
          if (cb.checked) {
            enabledTools.push(cb.value);
          }
        });
        
        voiceWs.send(JSON.stringify({ type: "set_tools", tools: enabledTools }));
        
      }
    }, 100);
  };

  voiceWs.onmessage = async (event) => {
    // Handle binary audio chunks first
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      const size = event.data instanceof Blob ? event.data.size : event.data.byteLength;
      log(`Received binary audio chunk: ${size} bytes`);
      await handleAudioChunk(event.data);
      return;
    }

    // Handle JSON messages
    if (typeof event.data === 'string') {
      try {
        const data = JSON.parse(event.data);
        log(`Received JSON message: ${data.type}`);
        await handleMessage(data);
      } catch (e) {
        log("Error parsing message: " + e + " Data: " + event.data.substring(0, 100));
      }
    } else {
      log("Unknown message type: " + typeof event.data + ", constructor: " + event.data.constructor.name);
    }
  };

  voiceWs.onerror = (e) => {
    clearTimeout(connectionTimeout);
    log("WebSocket error occurred - connection failed");
    console.error("WebSocket error details:", e);
    console.error("Error type:", e.type);
    console.error("Error target:", e.target);
    // Set status to disconnected on error
    setConnectionStatus("disconnected");
    voiceWs = null;
  };

  voiceWs.onclose = (event) => {
    clearTimeout(connectionTimeout);
    log(`Voice WebSocket closed - Code: ${event.code}, Reason: ${event.reason || 'none'}, WasClean: ${event.wasClean}`);
    console.log("Close event details:", event);
    setConnectionStatus("disconnected");
    pushToTalkBtn.disabled = true;
    voiceWs = null;

    // Only auto-reconnect if it wasn't a clean close (code 1000) or user-initiated disconnect
    // And only if we're not in the middle of a manual disconnect
    if (event.code !== 1000 && !isManualDisconnect) {
      log(`Auto-reconnect scheduled (code: ${event.code}, manual: ${isManualDisconnect})`);
      // Auto-reconnect after 2 seconds
      setTimeout(() => {
        if (!voiceWs || voiceWs.readyState === WebSocket.CLOSED) {
          log("Attempting to reconnect...");
          setConnectionStatus("connecting");
          connectVoiceWebSocket();
        }
      }, 2000);
    } else {
      log(`Not auto-reconnecting (code: ${event.code}, manual: ${isManualDisconnect})`);
    }
    isManualDisconnect = false;
  };
}

// Markdown Assistant state and functions
let markdownModalOpen = false;
let markdownContent = "";

function openMarkdownEditor(task) {
  const modal = document.getElementById("agentMarkdownEditorModal");
  const taskDesc = document.getElementById("markdownTaskDescription");
  const editor = document.getElementById("agentMarkdownEditor");
  const preview = document.getElementById("agentMarkdownPreview");
  
  markdownContent = "";
  editor.innerHTML = '<span style="color: #6a9955;">// Waiting for agent to start writing...</span>';
  preview.innerHTML = '<span style="color: #adb5bd; font-style: italic;">Preview will appear here...</span>';
  if (taskDesc) taskDesc.textContent = task || "Working on your document...";
  
  // Reset status
  const statusEl = document.getElementById("markdownStatus");
  if (statusEl) {
    statusEl.innerHTML = `<span style="width: 8px; height: 8px; background: #3498db; border-radius: 50%; animation: pulse 2s infinite;"></span><span style="color: #666;">Agent is working...</span>`;
  }
  
  modal.style.display = "block";
  markdownModalOpen = true;
  log(`Markdown editor opened: ${task}`);
}

function closeMarkdownEditor() {
  const modal = document.getElementById("agentMarkdownEditorModal");
  modal.style.display = "none";
  markdownModalOpen = false;
  markdownContent = "";
  log("Markdown editor closed");
}

// ========================================
// HTML Editor Functions
// ========================================

let htmlModalOpen = false;
let htmlContent = "";

function openHtmlEditor(task) {
  const modal = document.getElementById("agentHtmlEditorModal");
  const taskDesc = document.getElementById("htmlTaskDescription");
  const editor = document.getElementById("agentHtmlEditor");
  const preview = document.getElementById("agentHtmlPreview");
  const charCount = document.getElementById("htmlCharCount");
  
  htmlContent = "";
  editor.innerHTML = '<span style="color: #6a9955;">// Waiting for agent to start writing...</span>';
  preview.srcdoc = '<html><body style="font-family: sans-serif; color: #adb5bd; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0;"><p>Preview will appear here...</p></body></html>';
  if (taskDesc) taskDesc.textContent = task || "Building your webpage...";
  if (charCount) charCount.textContent = "0 chars";
  
  // Reset status
  const statusEl = document.getElementById("htmlStatus");
  if (statusEl) {
    statusEl.innerHTML = `<span style="width: 8px; height: 8px; background: #3498db; border-radius: 50%; animation: pulse 2s infinite;"></span><span style="color: #666;">Agent is building...</span>`;
  }
  
  modal.style.display = "block";
  htmlModalOpen = true;
  log(`HTML editor opened: ${task}`);
}

function closeHtmlEditor() {
  const modal = document.getElementById("agentHtmlEditorModal");
  modal.style.display = "none";
  htmlModalOpen = false;
  htmlContent = "";
  log("HTML editor closed");
}

function refreshHtmlPreview() {
  const preview = document.getElementById("agentHtmlPreview");
  if (preview && htmlContent) {
    preview.srcdoc = htmlContent;
    log("HTML preview refreshed");
  }
}

function copyHtmlCode() {
  if (htmlContent) {
    navigator.clipboard.writeText(htmlContent).then(() => {
      const btn = document.getElementById("copyHtmlBtn");
      const original = btn.textContent;
      btn.textContent = "✓ Copied!";
      setTimeout(() => btn.textContent = original, 2000);
    });
  }
}

function downloadHtml() {
  if (htmlContent) {
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated-page.html';
    a.click();
    URL.revokeObjectURL(url);
    log("HTML downloaded");
  }
}

// ========================================
// Inline Reasoning Display (Nemotron)
// ========================================

let currentReasoningContainer = null;
let currentThinkingEl = null;
let currentConclusionEl = null;
let inlineThinkingContent = "";
let inlineConclusionContent = "";

function createReasoningMessage(problem) {
  // Create a special reasoning message in the chat
  const container = document.createElement("div");
  container.className = "message assistant reasoning-message";
  container.style.cssText = "background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #76b900; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;";
  
  // Header
  const header = document.createElement("div");
  header.style.cssText = "display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(118, 185, 0, 0.3);";
  header.innerHTML = `<span style="font-size: 1.2rem;">🧠</span><span style="color: #76b900; font-weight: 600;">Nemotron Reasoning</span><span id="reasoningSpinner" style="margin-left: auto; width: 12px; height: 12px; border: 2px solid #76b900; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite;"></span>`;
  container.appendChild(header);
  
  // Thinking section (collapsible)
  const thinkingSection = document.createElement("div");
  thinkingSection.style.cssText = "margin-bottom: 0.75rem;";
  
  const thinkingHeader = document.createElement("div");
  thinkingHeader.style.cssText = "display: flex; align-items: center; gap: 0.5rem; cursor: pointer; color: #8b949e; font-size: 0.85rem;";
  thinkingHeader.innerHTML = `<span>💭</span><span>Thinking...</span>`;
  thinkingSection.appendChild(thinkingHeader);
  
  const thinkingContent = document.createElement("div");
  thinkingContent.id = "inlineThinkingContent";
  thinkingContent.style.cssText = "color: #8b949e; font-size: 0.9rem; line-height: 1.5; margin-top: 0.5rem; max-height: 300px; overflow-y: auto; font-family: 'Monaco', 'Menlo', monospace; white-space: pre-wrap; padding-right: 0.5rem;";
  thinkingSection.appendChild(thinkingContent);
  container.appendChild(thinkingSection);
  
  // Conclusion section
  const conclusionSection = document.createElement("div");
  conclusionSection.id = "inlineConclusionSection";
  conclusionSection.style.cssText = "display: none;";
  
  const conclusionHeader = document.createElement("div");
  conclusionHeader.style.cssText = "display: flex; align-items: center; gap: 0.5rem; color: #76b900; font-size: 0.85rem; margin-bottom: 0.5rem;";
  conclusionHeader.innerHTML = `<span>✨</span><span>Conclusion</span>`;
  conclusionSection.appendChild(conclusionHeader);
  
  const conclusionContent = document.createElement("div");
  conclusionContent.id = "inlineConclusionContent";
  conclusionContent.style.cssText = "color: #e0e0e0; font-size: 0.95rem; line-height: 1.6;";
  conclusionSection.appendChild(conclusionContent);
  container.appendChild(conclusionSection);
  
  getActiveConversationEl().appendChild(container);
  scrollToBottom();
  
  currentReasoningContainer = container;
  currentThinkingEl = thinkingContent;
  currentConclusionEl = conclusionContent;
  inlineThinkingContent = "";
  inlineConclusionContent = "";
}

function appendInlineThinking(content) {
  if (!currentThinkingEl) return;
  inlineThinkingContent += content;
  currentThinkingEl.textContent = inlineThinkingContent;
  currentThinkingEl.scrollTop = currentThinkingEl.scrollHeight;
  
  // Also scroll the main conversation to keep the reasoning card visible
  const activeEl = getActiveConversationEl();
  if (activeEl) {
    activeEl.scrollTop = activeEl.scrollHeight;
  }
}

function appendInlineConclusion(content) {
  if (!currentConclusionEl) return;
  
  // Show conclusion section on first content
  const section = document.getElementById("inlineConclusionSection");
  if (section && section.style.display === "none") {
    section.style.display = "block";
  }
  
  inlineConclusionContent += content;
  currentConclusionEl.textContent = inlineConclusionContent;
  
  // Also scroll the main conversation to keep the reasoning card visible
  const activeEl = getActiveConversationEl();
  if (activeEl) {
    activeEl.scrollTop = activeEl.scrollHeight;
  }
}

function finalizeReasoningMessage(finalContent) {
  // Hide spinner
  const spinner = document.getElementById("reasoningSpinner");
  if (spinner) {
    spinner.style.display = "none";
  }
  
  // If we have conclusion content, make sure it's shown
  if (finalContent && currentConclusionEl) {
    const section = document.getElementById("inlineConclusionSection");
    if (section) section.style.display = "block";
    currentConclusionEl.textContent = finalContent;
  }
  
  // Collapse thinking section if there's a conclusion
  if (inlineConclusionContent && currentThinkingEl) {
    const thinkingSection = currentThinkingEl.parentElement;
    if (thinkingSection) {
      thinkingSection.style.maxHeight = "100px";
      thinkingSection.style.overflow = "hidden";
      thinkingSection.style.cursor = "pointer";
      thinkingSection.title = "Click to expand thinking";
      thinkingSection.onclick = () => {
        if (thinkingSection.style.maxHeight === "100px") {
          thinkingSection.style.maxHeight = "400px";
        } else {
          thinkingSection.style.maxHeight = "100px";
        }
      };
    }
  }
  
  currentReasoningContainer = null;
  currentThinkingEl = null;
  currentConclusionEl = null;
}

// ========================================
// Reasoning Modal Functions (Nemotron) - Legacy
// ========================================

let reasoningModalOpen = false;
let reasoningThinking = "";
let reasoningConclusion = "";

function openReasoningModal(task, analysisType) {
  const modal = document.getElementById("agentReasoningModal");
  const taskDesc = document.getElementById("reasoningTaskDescription");
  const thinkingEl = document.getElementById("agentReasoningThinking");
  const conclusionEl = document.getElementById("agentReasoningConclusion");
  
  reasoningThinking = "";
  reasoningConclusion = "";
  
  thinkingEl.innerHTML = '<span style="color: #6a9955;">// Reasoning process will appear here...</span>';
  conclusionEl.innerHTML = '<span style="color: #adb5bd; font-style: italic;">Analysis results will appear here...</span>';
  
  // Format task description
  let desc = task || "Analyzing...";
  if (analysisType && analysisType !== "general") {
    desc = `[${analysisType}] ${desc}`;
  }
  if (taskDesc) taskDesc.textContent = desc.length > 100 ? desc.substring(0, 100) + "..." : desc;
  
  // Reset status
  const statusEl = document.getElementById("reasoningStatus");
  if (statusEl) {
    statusEl.innerHTML = `<span style="width: 8px; height: 8px; background: #76b900; border-radius: 50%; animation: pulse 1s infinite;"></span><span style="color: #666;">Nemotron is thinking...</span>`;
  }
  
  // Show thinking indicator
  const thinkingIndicator = document.getElementById("thinkingIndicator");
  if (thinkingIndicator) thinkingIndicator.style.display = "block";
  
  modal.style.display = "block";
  reasoningModalOpen = true;
  log(`Reasoning modal opened: ${task}`);
}

function closeReasoningModal() {
  const modal = document.getElementById("agentReasoningModal");
  modal.style.display = "none";
  reasoningModalOpen = false;
  reasoningThinking = "";
  reasoningConclusion = "";
  log("Reasoning modal closed");
}

function appendReasoningThinking(content) {
  if (!reasoningModalOpen) return;
  
  const thinkingEl = document.getElementById("agentReasoningThinking");
  if (!thinkingEl) return;
  
  // Clear placeholder on first content
  if (reasoningThinking === "") {
    thinkingEl.innerHTML = "";
  }
  
  reasoningThinking += content;
  
  // Escape HTML and preserve newlines
  const escaped = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  thinkingEl.innerHTML += escaped;
  
  // Auto-scroll
  thinkingEl.scrollTop = thinkingEl.scrollHeight;
}

function appendReasoningConclusion(content) {
  if (!reasoningModalOpen) return;
  
  const conclusionEl = document.getElementById("agentReasoningConclusion");
  if (!conclusionEl) return;
  
  // Clear placeholder on first content
  if (reasoningConclusion === "") {
    conclusionEl.innerHTML = "";
  }
  
  reasoningConclusion += content;
  
  // Render as markdown
  renderMarkdownPreview(reasoningConclusion, "agentReasoningConclusion");
  
  // Auto-scroll
  conclusionEl.scrollTop = conclusionEl.scrollHeight;
}

function completeReasoning() {
  const statusEl = document.getElementById("reasoningStatus");
  if (statusEl) {
    statusEl.innerHTML = `<span style="width: 8px; height: 8px; background: #28a745; border-radius: 50%;"></span><span style="color: #28a745; font-weight: 600;">Analysis complete</span>`;
  }
  
  // Hide thinking indicator
  const thinkingIndicator = document.getElementById("thinkingIndicator");
  if (thinkingIndicator) thinkingIndicator.style.display = "none";
  
  log("Reasoning complete");
}

function copyReasoningOutput() {
  const output = `## Thinking\n${reasoningThinking}\n\n## Conclusion\n${reasoningConclusion}`;
  navigator.clipboard.writeText(output).then(() => {
    const btn = document.getElementById("copyReasoningBtn");
    if (btn) {
      const original = btn.textContent;
      btn.textContent = "✓ Copied!";
      setTimeout(() => btn.textContent = original, 2000);
    }
  });
}

function renderMarkdownPreview(markdown, containerId) {
  // Use marked.js for proper markdown rendering
  if (typeof marked !== 'undefined') {
    // Configure marked for GFM (GitHub Flavored Markdown) with tables
    marked.setOptions({
      gfm: true,
      breaks: true,
      tables: true,
      sanitize: false
    });
    
    // Custom renderer for mermaid code blocks
    const renderer = new marked.Renderer();
    const originalCodeRenderer = renderer.code;
    
    renderer.code = function(code, language) {
      // Handle both old and new marked.js API
      const codeText = typeof code === 'object' ? code.text : code;
      const codeLang = typeof code === 'object' ? code.lang : language;
      
      if (codeLang === 'mermaid') {
        // Return a placeholder div for mermaid to process
        const mermaidId = 'mermaid-' + Math.random().toString(36).substr(2, 9);
        return `<div class="mermaid" id="${mermaidId}">${codeText}</div>`;
      }
      // For other code blocks, use default rendering
      return `<pre><code class="language-${codeLang || ''}">${escapeHtml(codeText)}</code></pre>`;
    };
    
    marked.setOptions({ renderer: renderer });
    
    let html = marked.parse(markdown);
    
    // Schedule mermaid rendering after DOM update
    if (containerId && markdown.includes('```mermaid')) {
      setTimeout(() => {
        try {
          if (typeof mermaid !== 'undefined') {
            mermaid.init(undefined, document.querySelectorAll(`#${containerId} .mermaid`));
          }
        } catch (e) {
          console.log('Mermaid rendering error:', e);
        }
      }, 100);
    }
    
    return html;
  }
  
  // Fallback: Simple markdown to HTML converter if marked.js not loaded
  let html = markdown
    // Code blocks first (before other formatting)
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Tables (basic GFM table support)
    .replace(/^\|(.+)\|$/gm, function(match, content) {
      const cells = content.split('|').map(c => c.trim());
      return '<tr>' + cells.map(c => {
        if (c.match(/^[-:]+$/)) return ''; // Skip separator row
        return '<td>' + c + '</td>';
      }).join('') + '</tr>';
    })
    // Headers
    .replace(/^### (.*)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*)$/gm, '<h2>$1</h2>')
    .replace(/^# (.*)$/gm, '<h1>$1</h1>')
    // Bold and italic
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
    // Blockquotes
    .replace(/^> (.*)$/gm, '<blockquote>$1</blockquote>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr>')
    // Unordered lists
    .replace(/^- (.*)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)\n(?=<li>)/g, '$1')
    // Line breaks to paragraphs (simple approach)
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  
  // Wrap in paragraph if not starting with block element
  if (!html.startsWith('<h') && !html.startsWith('<pre') && !html.startsWith('<ul') && !html.startsWith('<ol') && !html.startsWith('<blockquote') && !html.startsWith('<table')) {
    html = '<p>' + html + '</p>';
  }
  
  return html;
}

function copyMarkdownToClipboard() {
  navigator.clipboard.writeText(markdownContent).then(() => {
    const btn = document.getElementById("copyMarkdownBtn");
    const originalText = btn.textContent;
    btn.textContent = "✓ Copied!";
    setTimeout(() => { btn.textContent = originalText; }, 2000);
  });
}

// Close button handler
document.addEventListener("DOMContentLoaded", () => {
  // Markdown modal handlers
  const closeMarkdownBtn = document.getElementById("closeMarkdownModal");
  if (closeMarkdownBtn) {
    closeMarkdownBtn.onclick = closeMarkdownEditor;
  }

  const copyMarkdownBtn = document.getElementById("copyMarkdownBtn");
  if (copyMarkdownBtn) {
    copyMarkdownBtn.onclick = copyMarkdownToClipboard;
  }

  // Close on escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (markdownModalOpen) closeMarkdownEditor();
    }
  });
});

async function handleMessage(data) {
  log(`Received message: ${data.type}`);

  switch (data.type) {
    case "connected":
      log("Server ready - connection established");
      // Initialize audio context when connected (needed for greeting playback)
      initializeAudioOnConnection();
      // Greeting will be sent by server automatically
      break;

    case "asr_partial":
      // Update current user message with partial transcription
      removeEmptyState();
      if (!currentUserMsg) {
        // Create new user message for this recording session
        const msg = createMessageElement("user", "");
        getActiveConversationEl().appendChild(msg.container);
        currentUserMsg = msg;
      }
      currentUserMsg.content.textContent = data.text;
      scrollToBottom();
      break;

    case "asr_final":
      // Final transcription received
      removeEmptyState();
      if (!currentUserMsg) {
        // No current message - only create one if we're NOT already recording again
        // (If isRecording is true, this is a stale asr_final from previous recording)
        if (!isRecording) {
          const msg = createMessageElement("user", data.text);
          getActiveConversationEl().appendChild(msg.container);
          currentUserMsg = msg;
        }
      } else {
        currentUserMsg.content.textContent = data.text;
      }
      // Clear current user message reference so next recording creates a new one
      currentUserMsg = null;
      log(`Final transcription: "${data.text}"`);

      // Show thinking indicator
      showThinkingIndicator();

      // Update voice/video call status
      if (voiceCallActive) {
        updateVoiceCallStatus('processing', 'Thinking...');
      }
      if (videoCallActive) {
        updateVideoCallStatus('processing', 'Looking & thinking...');
      }
      break;

    case "transient_response":
      // Transient response (e.g., "on it", "thinking...")
      hideThinkingIndicator();
      removeEmptyState();
      if (currentTransientMsg) {
        currentTransientMsg.container.remove();
      }
      const transientMsg = createMessageElement("assistant", data.text, true);
      getActiveConversationEl().appendChild(transientMsg.container);
      currentTransientMsg = transientMsg;
      scrollToBottom();
      log(`Transient response: "${data.text}"`);
      break;

    case "agent_started":
      // Agent tool was called - open appropriate UI
      hideThinkingIndicator();
      log(`Agent started: ${data.agent_type}, task: ${data.task ? data.task.substring(0, 50) : 'none'}...`);
      if (data.agent_type === "markdown_assistant") {
        openMarkdownEditor(data.task);
      } else if (data.agent_type === "html_assistant") {
        openHtmlEditor(data.task);
      } else if (data.agent_type === "reasoning_assistant") {
        // Reasoning now uses inline display, not popup
        log("Reasoning uses inline display");
      }
      break;

    case "agent_markdown_chunk":
      // Stream markdown to editor
      if (markdownModalOpen) {
        const mdEditor = document.getElementById("agentMarkdownEditor");
        const mdPreview = document.getElementById("agentMarkdownPreview");
        
        if (data.done) {
          // Agent finished
          const statusEl = document.getElementById("markdownStatus");
          if (statusEl) {
            statusEl.innerHTML = '<span style="width: 8px; height: 8px; background: #2ecc71; border-radius: 50%;"></span><span style="color: #666;">Document complete</span>';
          }
        } else {
          const content = data.content || "";
          markdownContent += content;
          
          // Update raw markdown view
          if (mdEditor) {
            mdEditor.textContent = markdownContent;
            mdEditor.scrollTop = mdEditor.scrollHeight;
          }
          
          // Update preview
          if (mdPreview) {
            mdPreview.innerHTML = renderMarkdownPreview(markdownContent, 'agentMarkdownPreview');
            mdPreview.scrollTop = mdPreview.scrollHeight;
          }
        }
      }
      break;

    case "agent_html_chunk":
      // Stream HTML to editor with live preview
      if (htmlModalOpen) {
        const htmlEditor = document.getElementById("agentHtmlEditor");
        const htmlPreview = document.getElementById("agentHtmlPreview");
        const htmlCharCount = document.getElementById("htmlCharCount");
        
        if (data.done) {
          // Agent finished - final preview update
          const statusEl = document.getElementById("htmlStatus");
          if (statusEl) {
            statusEl.innerHTML = '<span style="width: 8px; height: 8px; background: #2ecc71; border-radius: 50%;"></span><span style="color: #666;">Page complete!</span>';
          }
          // Final preview refresh
          if (htmlPreview && htmlContent) {
            htmlPreview.srcdoc = htmlContent;
          }
        } else {
          const content = data.content || "";
          htmlContent += content;
          
          // Update code view
          if (htmlEditor) {
            htmlEditor.textContent = htmlContent;
            htmlEditor.scrollTop = htmlEditor.scrollHeight;
          }
          
          // Update char count
          if (htmlCharCount) {
            htmlCharCount.textContent = `${htmlContent.length} chars`;
          }
          
          // Update live preview periodically (every 500 chars or when we have complete HTML structure)
          if (htmlPreview && htmlContent.length > 0) {
            // Only update preview if we have a reasonable amount of content
            if (htmlContent.includes('</body>') || htmlContent.includes('</html>') || htmlContent.length % 500 < content.length) {
              htmlPreview.srcdoc = htmlContent;
            }
          }
        }
      }
      break;

    case "agent_html_complete":
      // HTML generation complete
      log(`HTML assistant completed: ${data.task}`);
      if (htmlModalOpen) {
        const htmlPreview = document.getElementById("agentHtmlPreview");
        if (htmlPreview && data.html) {
          htmlContent = data.html;
          htmlPreview.srcdoc = data.html;
        }
      }
      break;
    
    case "agent_markdown_complete":
      // Markdown assistant finished - add to conversation
      removeEmptyState();
      const mdMsg = createMarkdownMessageElement(data.task, data.markdown);
      getActiveConversationEl().appendChild(mdMsg.container);
      scrollToBottom();
      saveCurrentChat();
      log(`Markdown assistant completed: ${data.task.substring(0, 50)}...`);
      break;

    case "reasoning_started":
      // Nemotron reasoning started - create inline thinking display
      hideThinkingIndicator();
      removeEmptyState();
      log(`Reasoning started: ${data.problem ? data.problem.substring(0, 50) : 'unknown'}...`);
      createReasoningMessage(data.problem);
      break;

    case "reasoning_thinking":
      // Stream thinking content inline
      if (data.content) {
        appendInlineThinking(data.content);
      }
      break;

    case "reasoning_content":
      // Stream conclusion content inline
      if (data.content) {
        appendInlineConclusion(data.content);
      }
      break;

    case "reasoning_complete":
      // Reasoning finished
      log(`Reasoning complete: ${data.thinking?.length || 0} thinking, ${data.conclusion?.length || 0} conclusion`);
      finalizeReasoningMessage(data.conclusion || data.thinking);
      scrollToBottom();
      saveCurrentChat();
      break;

    // Legacy handlers for popup (keeping for compatibility)
    case "agent_reasoning_thinking":
      if (data.content && !data.done) {
        appendReasoningThinking(data.content);
      }
      break;

    case "agent_reasoning_chunk":
      if (data.content && !data.done) {
        appendReasoningConclusion(data.content);
      }
      break;

    case "agent_reasoning_complete":
      completeReasoning();
      break;

    case "final_response":
      // Final response received
      hideThinkingIndicator();
      removeEmptyState();
      if (currentTransientMsg) {
        currentTransientMsg.container.remove();
        currentTransientMsg = null;
      }
      const finalMsg = createMessageElement("assistant", data.text);
      getActiveConversationEl().appendChild(finalMsg.container);
      scrollToBottom();
      // Save chat after assistant response
      saveCurrentChat();
      log(`Final response: "${data.text}"`);
      break;

    case "asr_result":
      // ASR final transcription result (used by video/voice call mode)
      removeEmptyState();
      const asrText = data.text || data.content || "";
      if (asrText.trim()) {
        if (currentUserMsg) {
          // Update existing message from asr_partial
          currentUserMsg.content.textContent = asrText;
        } else {
          // Create new message if none exists
          const userMsg = createMessageElement("user", asrText);
          getActiveConversationEl().appendChild(userMsg.container);
        }
        scrollToBottom();
        saveCurrentChat();
        log(`ASR result displayed: ${asrText}`);

        // Show thinking indicator after user's speech is displayed
        showThinkingIndicator();
      }
      // Clear currentUserMsg so next speech creates new message
      currentUserMsg = null;
      break;

    case "llm_final":
      // VLM/LLM final response (used by video call mode)
      hideThinkingIndicator();
      removeEmptyState();
      if (currentTransientMsg) {
        currentTransientMsg.container.remove();
        currentTransientMsg = null;
      }
      const llmText = data.text || data.content || "";
      if (llmText.trim()) {
        const vlmMsg = createMessageElement("assistant", llmText);
        getActiveConversationEl().appendChild(vlmMsg.container);
        scrollToBottom();
        saveCurrentChat();
        log(`VLM response: "${llmText.substring(0, 50)}..."`);
      }
      break;

    case "tts_start":
      // TTS streaming started
      log(`TTS started (transient: ${data.is_transient})`);
      isTtsPlaying = true;
      ttsAborted = false; // Reset abort flag for new TTS
      
      // PAUSE VAD during TTS to prevent echo detection
      if (videoCallActive && videoCallVadInstance && !pttMode) {
        try {
          videoCallVadInstance.pause();
          log('VAD paused during TTS playback');
        } catch (e) {}
      }
      
      // Ensure gain is restored
      if (masterGainNode && audioContext) {
        masterGainNode.gain.setValueAtTime(1, audioContext.currentTime);
      }
      
      // Update voice/video call status
      if (voiceCallActive) {
        updateVoiceCallStatus('speaking', 'Speaking...');
      }
      if (videoCallActive) {
        updateVideoCallStatus('speaking', 'Speaking...');
      }
      
      if (!audioContext) {
        try {
          audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: data.sample_rate || 24000 });
          log(`Audio context created with sample rate: ${data.sample_rate || 24000}, state: ${audioContext.state}`);
        } catch (e) {
          log("Error creating audio context: " + e);
          break;
        }
      }
      
      // Resume audio context if suspended (browser autoplay policy)
      if (audioContext.state === 'suspended') {
        audioContext.resume().then(() => {
          log(`Audio context resumed from suspended state, new state: ${audioContext.state}`);
        }).catch(e => {
          log("Error resuming audio context: " + e);
        });
      }
      
      // Reset nextPlayTime for new TTS stream - start immediately
      nextPlayTime = audioContext.currentTime;
      log(`TTS playback scheduled to start at: ${nextPlayTime.toFixed(3)}s, context state: ${audioContext.state}`);
      break;

    case "tts_done":
      // TTS streaming complete (server done sending chunks)
      // But audio may still be playing in the browser!
      log(`TTS done from server (transient: ${data.is_transient}), active sources: ${activeAudioSources.length}`);
      
      // Only set isTtsPlaying=false if no audio sources are still playing
      // Otherwise, the source.onended handler will set it to false
      if (activeAudioSources.length === 0) {
        isTtsPlaying = false;
        videoCallProcessing = false; // Clear processing flag
        lastTtsEndTime = Date.now(); // Track when TTS finished for cooldown
        
        // RESUME VAD after TTS playback ends (with delay for cooldown)
        if (videoCallActive && videoCallVadInstance && !pttMode) {
          setTimeout(() => {
            if (!isTtsPlaying && videoCallVadInstance) {
              try {
                videoCallVadInstance.start();
                log('VAD resumed after TTS playback');
              } catch (e) {}
            }
          }, 500); // 500ms delay before resuming VAD
        }
        
        // Update voice/video call status - back to listening
        if (voiceCallActive && !voiceCallMuted) {
          updateVoiceCallStatus('listening', 'Listening...');
        }
        if (videoCallActive && !videoCallMuted) {
          // Show appropriate status based on input mode
          if (pttMode) {
            updateVideoCallStatus('listening', 'Press SPACE or hold button to talk');
          } else {
            updateVideoCallStatus('listening', 'Listening...');
          }
        }
      } else {
        log(`Audio still playing (${activeAudioSources.length} sources), keeping isTtsPlaying=true`);
      }
      break;

    case "error":
      log("Error: " + data.error);
      alert("Error: " + data.error);
      // Clear processing flag and resume VAD on error
      videoCallProcessing = false;
      if (videoCallActive && videoCallVadInstance && !pttMode && !isTtsPlaying) {
        setTimeout(() => {
          try {
            videoCallVadInstance.start();
            log('VAD resumed after error');
            updateVideoCallStatus('listening', 'Listening...');
          } catch (e) {}
        }, 500);
      }
      break;

    case "pong":
      // Keep-alive response
      break;

    case "reset_ack":
      log("Conversation reset");
      break;

    case "voice_changed":
      log(`Voice changed to: ${data.voice}`);
      break;


    case "system_prompt":
      // Received current system prompt
      systemPromptInput.value = data.prompt || "";
      break;

    case "system_prompt_changed":
      log(`System prompt changed`);
      break;

    case "disconnect_ack":
      log("Disconnect acknowledged by server");
      break;
  }
}

async function handleAudioChunk(data) {
  // Check if TTS was aborted (barge-in) - discard incoming audio
  if (ttsAborted) {
    log('Audio chunk discarded (TTS aborted)');
    return;
  }
  
  // Initialize audio context if not already done
  if (!audioContext) {
    log("Initializing audio context for first chunk");
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      nextPlayTime = audioContext.currentTime;
      log(`Audio context created, state: ${audioContext.state}`);
      
      // Resume audio context if suspended (browser autoplay policy)
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
        log(`Audio context resumed, new state: ${audioContext.state}`);
      }
    } catch (e) {
      log("Error initializing audio context: " + e);
      return;
    }
  }

  // Ensure audio context exists and is running
  if (!audioContext) {
    initializeAudioContext();
  }
  
  if (audioContext && audioContext.state === 'suspended') {
    try {
      await audioContext.resume();
      log(`Audio context resumed from suspended state`);
    } catch (e) {
      // Browser autoplay policy may prevent this - user needs to interact first
      log("Audio context resume blocked (user interaction required)");
      // Don't throw - we'll try again on next chunk after user interaction
      return;
    }
  }

  const arrayBuffer = data instanceof Blob ? await data.arrayBuffer() : data;
  
  if (arrayBuffer.byteLength === 0) {
    log("Empty audio chunk received, skipping");
    return;
  }
  
  const int16Data = new Int16Array(arrayBuffer);
  
  if (int16Data.length === 0) {
    log("Empty int16 array, skipping");
    return;
  }
  
  // Convert Int16 to Float32
  const float32Data = new Float32Array(int16Data.length);
  for (let i = 0; i < float32Data.length; i++) {
    float32Data[i] = int16Data[i] / 32768.0;
  }
  
  // Create buffer and play
  try {
    const buffer = audioContext.createBuffer(1, float32Data.length, 24000);
    buffer.copyToChannel(float32Data, 0);
    
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    
    // Use master gain node for instant muting (barge-in)
    if (!masterGainNode) {
      masterGainNode = audioContext.createGain();
      masterGainNode.connect(audioContext.destination);
    }
    source.connect(masterGainNode);
    
    // Track this source for potential barge-in stop
    activeAudioSources.push(source);
    source.onended = () => {
      const idx = activeAudioSources.indexOf(source);
      if (idx > -1) activeAudioSources.splice(idx, 1);
      
      // When last audio source ends, TTS is truly done playing
      if (activeAudioSources.length === 0) {
        isTtsPlaying = false;
        videoCallProcessing = false; // Clear processing flag
        lastTtsEndTime = Date.now(); // Track when TTS finished for cooldown
        log('All audio sources finished playing');
        
        // RESUME VAD after TTS playback ends (with delay for cooldown)
        if (videoCallActive && videoCallVadInstance && !pttMode) {
          setTimeout(() => {
            if (!isTtsPlaying && videoCallVadInstance) {
              try {
                videoCallVadInstance.start();
                log('VAD resumed after audio sources finished');
              } catch (e) {}
            }
          }, 500); // 500ms delay before resuming VAD
        }
      }
    };
    
    // Schedule playback
    if (nextPlayTime === null || nextPlayTime < audioContext.currentTime) {
      nextPlayTime = audioContext.currentTime;
    }
    
    const playTime = nextPlayTime;
    source.start(playTime);
    nextPlayTime = playTime + buffer.duration;
    
    log(`Playing audio chunk: ${buffer.duration.toFixed(3)}s at ${playTime.toFixed(3)}s (${int16Data.length} samples), context state: ${audioContext.state}`);
  } catch (e) {
    log("Error playing audio chunk: " + e + ", audio context state: " + (audioContext ? audioContext.state : 'null'));
    console.error("Audio playback error:", e);
  }
}

async function startRecording() {
  if (isRecording) {
    log("Already recording, ignoring start request");
    return;
  }
  
  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    log(`Cannot start recording: WebSocket not ready (state: ${voiceWs ? voiceWs.readyState : 'null'})`);
    alert("Not connected. Please wait for connection.");
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Microphone access not available. Use https or localhost.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000
      } 
    });

    // Try to find a supported codec
    const codecs = [
      'audio/webm;codecs=opus',
      'audio/webm;codecs=vorbis',
      'audio/webm',
    ];
    
    let selectedMimeType = null;
    for (const codec of codecs) {
      if (MediaRecorder.isTypeSupported(codec)) {
        selectedMimeType = codec;
        break;
      }
    }

    const options = selectedMimeType ? { mimeType: selectedMimeType } : {};
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0 && voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(e.data);
      }
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(track => track.stop());
      // Send ASR end signal
      if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(JSON.stringify({ type: "asr_end" }));
      }
    };

    // Reset current user message BEFORE starting recording to prevent race condition
    // where asr_partial arrives before this line executes
    currentUserMsg = null;
    isRecording = true;
    mediaRecorder.start(150); // 150ms chunks for lower ASR latency
    pushToTalkBtn.classList.add("recording");
    pushToTalkBtn.disabled = false;
    log("Recording started");
  } catch (err) {
    log("Could not start recording: " + err);
    alert("Recording failed: " + err.message);
  }
}

function stopRecording() {
  if (!isRecording) {
    log("Not recording, ignoring stop request");
    return;
  }

  log("Stopping recording...");
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  isRecording = false;
  pushToTalkBtn.classList.remove("recording");
  pushToTalkBtn.disabled = false;
  log("Recording stopped");
}

function clearChat() {
  log("Clear Chat button clicked");
  
  // Clear regular conversation
  conversationEl.innerHTML = '<div class="empty-state">Connect and start a conversation</div>';
  
  // Also clear video conversation if it exists
  if (videoConversationEl) {
    videoConversationEl.innerHTML = '';
  }
  
  currentTransientMsg = null;
  currentReasoningContainer = null;
  currentThinkingEl = null;
  currentConclusionEl = null;
  
  // Clear current chat messages
  if (currentChatId && chats[currentChatId]) {
    chats[currentChatId].messages = [];
    chats[currentChatId].preview = '';
    saveChatsToStorage();
    renderChatList();
  }
  
  log("Chat cleared");
}

// Initialize
clearBtn.onclick = clearChat;

// Initialize chat system
loadChatsFromStorage();
if (!currentChatId || !chats[currentChatId]) {
  createNewChat();
} else {
  loadChat(currentChatId);
}

// Initialize text input handlers
if (textInput) {
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  });
  
  // Auto-resize textarea
  textInput.addEventListener('input', () => {
    textInput.style.height = 'auto';
    textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
  });
}

// Initialize text input state
updateTextInputState();

// Track manual disconnect to prevent auto-reconnect
let isManualDisconnect = false;

// Disconnect/Connect toggle handler
disconnectBtn.onclick = () => {
  log(`Disconnect/Connect button clicked. WebSocket state: ${voiceWs ? voiceWs.readyState : 'null'}`);
  
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    // Disconnect
    log("Manual disconnect initiated");
    isManualDisconnect = true;
    stopTtsPlayback(); // Stop any playing TTS immediately
    voiceWs.send(JSON.stringify({ type: "disconnect" }));
    log("Disconnecting...");
  } else {
    // Connect
    log("Connect button clicked - initiating connection");
    const prompt = systemPromptInput.value.trim();
    if (!prompt) {
      alert("Please enter a system prompt before connecting");
      log("No system prompt - connection cancelled");
      return;
    }
    
    // Store prompt to send after connection
    pendingSystemPrompt = prompt;
    log(`System prompt stored: ${prompt.substring(0, 50)}...`);
    
    // Connect WebSocket
    connectVoiceWebSocket();
  }
};

// Voice selection handler
voiceSelect.onchange = (e) => {
  const selectedVoice = e.target.value;
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: "set_voice", voice: selectedVoice }));
    log(`Voice selection changed to: ${selectedVoice}`);
  } else {
    log("Cannot change voice: WebSocket not connected");
  }
};

// Tool capability checkboxes handler
function updateEnabledTools() {
  const capabilityCheckboxes = document.querySelectorAll('input[id^="cap"]');
  const agentCheckboxes = document.querySelectorAll('input[id^="agent"]');
  const enabledTools = [];
  
  capabilityCheckboxes.forEach(cb => {
    if (cb.checked) {
      enabledTools.push(cb.value);
    }
  });
  
  agentCheckboxes.forEach(cb => {
    if (cb.checked) {
      enabledTools.push(cb.value);
    }
  });
  
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: "set_tools", tools: enabledTools }));
    log(`Tools updated: ${enabledTools.join(", ")}`);
  } else {
    log("Cannot update tools: WebSocket not connected");
  }
}

// Attach change handlers to all capability and agent checkboxes
document.addEventListener("DOMContentLoaded", () => {
  const capabilityCheckboxes = document.querySelectorAll('input[id^="cap"]');
  const agentCheckboxes = document.querySelectorAll('input[id^="agent"]');
  
  capabilityCheckboxes.forEach(cb => {
    cb.addEventListener("change", updateEnabledTools);
  });
  
  agentCheckboxes.forEach(cb => {
    cb.addEventListener("change", updateEnabledTools);
  });
});


// Store pending system prompt
let pendingSystemPrompt = null;

// Save prompt button (if it exists - kept for compatibility)
if (savePromptBtn) {
  savePromptBtn.onclick = () => {
    const prompt = systemPromptInput.value.trim();
    if (prompt && voiceWs && voiceWs.readyState === WebSocket.OPEN) {
      voiceWs.send(JSON.stringify({ type: "set_system_prompt", prompt: prompt }));
      log(`System prompt updated`);
    } else if (!prompt) {
      alert("Please enter a system prompt");
    } else {
      log("Cannot update prompt: WebSocket not connected");
    }
  };
}

// Push-to-talk button functionality (works for both touch and mouse)
let isPressingButton = false;

function handlePushStart(e) {
  e.preventDefault();
  e.stopPropagation();
  
  if (isRecording || !voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    log("Cannot start recording: " + (isRecording ? "already recording" : "not connected"));
    return;
  }
  
  isPressingButton = true;
  pushToTalkBtn.classList.add("recording");
  startRecording();
}

function handlePushEnd(e) {
  e.preventDefault();
  e.stopPropagation();
  
  if (isPressingButton && isRecording) {
    isPressingButton = false;
    pushToTalkBtn.classList.remove("recording");
    stopRecording();
  }
}

// Add event listeners to push-to-talk button
if (pushToTalkBtn) {
  // Mouse events
  pushToTalkBtn.addEventListener("mousedown", handlePushStart);
  pushToTalkBtn.addEventListener("mouseup", handlePushEnd);
  pushToTalkBtn.addEventListener("mouseleave", handlePushEnd); // Stop if mouse leaves button
  
  // Touch events (for mobile)
  pushToTalkBtn.addEventListener("touchstart", handlePushStart, { passive: false });
  pushToTalkBtn.addEventListener("touchend", handlePushEnd, { passive: false });
  pushToTalkBtn.addEventListener("touchcancel", handlePushEnd, { passive: false });
  
  log("Push-to-talk button event listeners attached");
} else {
  log("ERROR: pushToTalkBtn not found!");
}

// Hold-to-talk functionality - using 0 (zero) key (still works for keyboard users)
// Setup key listeners after page is fully loaded
function setupKeyListeners() {
  // Main recording handler
  window.addEventListener("keydown", function(e) {
    // Don't trigger recording if user is typing in an input/textarea
    const activeElement = document.activeElement;
    const isTyping = activeElement && (
      activeElement.tagName === "INPUT" || 
      activeElement.tagName === "TEXTAREA" ||
      (activeElement.isContentEditable && activeElement !== document.body)
    );
    
    // Check if the 0 (zero) key is pressed
    const isRecordKey = e.code === "Digit0" || 
                       e.key === "0" || 
                       e.keyCode === 48;
    
    if (isRecordKey && !isRecording && !e.repeat && !isTyping) {
      e.preventDefault();
      e.stopPropagation();
      pushToTalkBtn.classList.add("recording");
      startRecording();
    }
  }, { capture: true, passive: false });

  window.addEventListener("keyup", function(e) {
    // Check if the 0 (zero) key is released
    const isRecordKey = e.code === "Digit0" || 
                       e.key === "0" || 
                       e.keyCode === 48;
    
    if (isRecordKey) {
      e.preventDefault();
      e.stopPropagation();
      if (isRecording) {
        pushToTalkBtn.classList.remove("recording");
        stopRecording();
      }
    }
  }, { capture: true, passive: false });
}

// Setup key listeners when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupKeyListeners);
} else {
  // DOM is already ready
  setupKeyListeners();
}

// Initialize audio context on first user interaction (required by browsers)
let audioInitialized = false;

function initializeAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    nextPlayTime = audioContext.currentTime;
    audioInitialized = true;
    log("Audio context initialized");
  }
  
  // Resume if suspended
  if (audioContext.state === 'suspended') {
    return audioContext.resume().then(() => {
      log("Audio context resumed");
    }).catch((e) => {
      log("Error resuming audio context: " + e);
    });
  }
  
  return Promise.resolve();
}

function initializeAudioOnInteraction() {
  initializeAudioContext();
}

function initializeAudioOnConnection() {
  // Try to initialize audio context on connection
  // This may fail due to browser autoplay policy, but we try anyway
  initializeAudioContext().catch(() => {
    // If it fails, we'll initialize on first user interaction
    log("Audio context initialization deferred (will initialize on user interaction)");
  });
}

// Initialize audio on any user interaction (fallback)
document.addEventListener('click', initializeAudioOnInteraction, { once: true });
document.addEventListener('keydown', initializeAudioOnInteraction, { once: true });
document.addEventListener('touchstart', initializeAudioOnInteraction, { once: true });

// Auto-connect immediately on page load with default system prompt
// Run after all variables and functions are defined
console.log("[AUTO-CONNECT] Script loaded, setting up auto-connect...");

// Fetch default system prompt from server
async function loadDefaultSystemPrompt() {
  try {
    const response = await fetch('/api/default_prompt');
    if (response.ok) {
      const data = await response.json();
      if (data.prompt && systemPromptInput) {
        systemPromptInput.value = data.prompt;
        log(`Loaded default system prompt from server: ${data.prompt.substring(0, 50)}...`);
      }
    }
  } catch (err) {
    log(`Failed to load default prompt: ${err.message}`);
  }
}

// Use window.onload to ensure everything is ready
window.addEventListener('load', async () => {
  console.log("[INIT] Window loaded");
  log("=== INITIALIZING ===");
  
  // Load default system prompt from server if textarea is empty
  if (systemPromptInput && !systemPromptInput.value.trim()) {
    await loadDefaultSystemPrompt();
  }
  
  try {
    if (systemPromptInput) {
      pendingSystemPrompt = systemPromptInput.value.trim() || "";
      log(`System prompt loaded: ${pendingSystemPrompt.substring(0, 50)}...`);
    }
    
    // Don't auto-connect - wait for user to start a chat
    setConnectionStatus("disconnected");
    log("=== READY (waiting for user to start chat) ===");
  } catch (e) {
    log(`ERROR in init: ${e}`);
    console.error("[INIT] Error:", e);
    setConnectionStatus("disconnected");
  }
});

// Keep-alive ping every 30 seconds
setInterval(() => {
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: "ping" }));
  }
}, 30000);