/* ==========================================================================
   Spark Realtime — Audio, VAD, Push-to-Talk, Video Call & Face Recognition
   ========================================================================== */

let waveformBars = [];
let videoCallWaveformBars = [];
let bottomWaveformBars = [];

function calculateRmsEnergy(audioSamples) {
  if (!audioSamples || audioSamples.length === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < audioSamples.length; i++) {
    sumSquares += audioSamples[i] * audioSamples[i];
  }
  return Math.sqrt(sumSquares / audioSamples.length);
}

function hasEnoughEnergy(audioSamples, threshold = energyThreshold) {
  const energy = calculateRmsEnergy(audioSamples);
  return energy >= threshold;
}

function updateEnergyThreshold(value) {
  energyThreshold = parseFloat(value);
  ['energyThresholdValue', 'videoEnergyThresholdValue', 'settingsEnergyThresholdValue'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = energyThreshold.toFixed(3);
  });
  ['energyThreshold', 'videoEnergyThreshold', 'settingsEnergyThreshold'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = energyThreshold;
  });
  log(`Energy threshold updated to ${energyThreshold.toFixed(3)}`);
}

function updateVadSensitivity(value) {
  vadSensitivityValue = parseFloat(value);
  const el = document.getElementById('vadSensitivityValue');
  if (el) el.textContent = vadSensitivityValue.toFixed(2);
  log(`VAD sensitivity updated to: ${vadSensitivityValue}`);
}

function updateEnergyMeter(energy) {
  const percent = Math.min(100, (energy / 0.05) * 100);
  ['voiceEnergyBar', 'videoEnergyBar', 'bottomEnergyBar'].forEach((id) => {
    const bar = document.getElementById(id);
    if (bar) {
      bar.style.width = `${percent}%`;
      bar.style.background = energy < energyThreshold ? 'var(--text-muted)' : 'var(--accent)';
    }
  });
  ['voiceEnergyValue', 'videoEnergyValue', 'bottomEnergyValue'].forEach((id) => {
    const val = document.getElementById(id);
    if (val) val.textContent = energy.toFixed(3);
  });
}

function initAllWaveforms() {
  const buildBars = (containerId, count) => {
    const el = document.getElementById(containerId);
    if (!el) return [];
    el.innerHTML = '';
    const arr = [];
    for (let i = 0; i < count; i++) {
      const bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.height = '4px';
      el.appendChild(bar);
      arr.push(bar);
    }
    return arr;
  };
  waveformBars = buildBars('voiceWaveform', 18);
  videoCallWaveformBars = buildBars('videoCallWaveform', 16);
  bottomWaveformBars = buildBars('bottomWaveform', 12);
}

let waveformLoopRunning = false;
function startWaveformLoop() {
  if (waveformLoopRunning) return;
  waveformLoopRunning = true;

  const tick = () => {
    const activeSpeech = isRecording || pttRecording || isCurrentlySpeaking || videoCallSpeaking || currentSystemState === 'listening';
    const activeTts = isTtsPlaying || currentSystemState === 'speaking';
    const t = Date.now() / 1000;

    const updateSet = (bars, maxH, minH) => {
      bars.forEach((bar, i) => {
        let h = minH;
        if (activeSpeech) {
          h = Math.max(minH, Math.abs(Math.sin(t * 7 + i * 0.6)) * maxH * (0.55 + Math.random() * 0.45));
        } else if (activeTts) {
          h = Math.max(minH, (Math.sin(t * 5 + i * 0.45) * 0.5 + 0.5) * (maxH * 0.75));
        } else if (voiceCallActive || videoCallActive) {
          h = minH + (Math.sin(t * 2 + i * 0.35) * 0.5 + 0.5) * 4;
        }
        bar.style.height = `${Math.round(h)}px`;
      });
    };

    updateSet(waveformBars, 24, 4);
    updateSet(videoCallWaveformBars, 22, 4);
    updateSet(bottomWaveformBars, 15, 3);

    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function float32ToWav(float32Array, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32Array.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + float32Array.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, float32Array.length * 2, true);

  let offset = 44;
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function createLocalVadConfig(callbacks) {
  return Object.assign({
    workletURL: '/static/vendor/vad.worklet.bundle.min.js',
    modelURL: '/static/vendor/silero_vad.onnx',
    positiveSpeechThreshold: vadSensitivityValue || 0.5,
    negativeSpeechThreshold: 0.35,
    redemptionFrames: 8,
    preSpeechPadFrames: 10,
    minSpeechFrames: 3
  }, callbacks);
}

// ============================================================================
// Voice Call Mode (VAD)
// ============================================================================
async function setupVoiceCallMode() {
  log('Setting up Voice Call mode with local Silero VAD...');
  const container = document.getElementById('voiceCallContainer');
  if (container) container.classList.add('active');
  initAllWaveforms();
  startWaveformLoop();

  try {
    if (typeof vad === 'undefined') {
      throw new Error('Local VAD bundle not available');
    }
    vadInstance = await vad.MicVAD.new(createLocalVadConfig({
      onSpeechStart: () => {
        if (voiceCallMuted) return;
        if (isTtsPlaying && bargeInEnabled) {
          stopTtsPlayback();
        }
        currentUserMsg = null;
        isCurrentlySpeaking = true;
        updateVoiceCallStatus('listening', 'Listening to speech...');
      },
      onSpeechEnd: (audio) => {
        if (voiceCallMuted) return;
        const energy = calculateRmsEnergy(audio);
        updateEnergyMeter(energy);
        const now = Date.now();
        if (now - lastVadSendTime < VAD_DEBOUNCE_MS) return;
        lastVadSendTime = now;

        if (energy < energyThreshold) {
          isCurrentlySpeaking = false;
          updateVoiceCallStatus('listening', 'Listening...');
          return;
        }
        isCurrentlySpeaking = false;
        updateVoiceCallStatus('processing', 'Thinking...');
        sendVoiceCallAudio(audio);
      },
      onVADMisfire: () => {
        if (!isCurrentlySpeaking) updateVoiceCallStatus('listening', 'Listening...');
      }
    }));

    vadInstance.start();
    voiceCallActive = true;
    updateVoiceCallStatus('listening', 'Listening...');
  } catch (error) {
    log(`VAD initialization notice: ${error.message}`);
    voiceCallActive = true;
    updateVoiceCallStatus('listening', 'Hold Space or Mic to talk');
  }
}

function updateVoiceCallStatus(state, text) {
  const statusEl = document.getElementById('voiceCallStatus');
  const textEl = document.getElementById('voiceCallStatusText');
  if (statusEl) statusEl.className = `voice-call-status ${state}`;
  if (textEl) textEl.textContent = text;

  if (state === 'listening' || state === 'hearing') {
    setSystemState('listening', text);
  } else if (state === 'processing') {
    setSystemState('thinking', text);
  } else if (state === 'speaking') {
    setSystemState('speaking', text);
  }
}

async function sendVoiceCallAudio(audioFloat32) {
  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    updateVoiceCallStatus('listening', 'Disconnected');
    return;
  }
  const wavBlob = float32ToWav(audioFloat32, 16000);
  const reader = new FileReader();
  reader.onload = () => {
    const base64 = reader.result.split(',')[1];
    voiceWs.send(JSON.stringify({ type: 'asr_audio', audio: base64, format: 'wav' }));
  };
  reader.readAsDataURL(wavBlob);
}

function toggleVoiceCallMute() {
  voiceCallMuted = !voiceCallMuted;
  const btn = document.getElementById('voiceCallMuteBtn');
  if (btn) {
    btn.classList.toggle('muted', voiceCallMuted);
    btn.innerHTML = `${window.SparkIcons.iconSvg(voiceCallMuted ? 'mic-off' : 'mic', 14)}<span>${voiceCallMuted ? 'Unmute' : 'Mute'}</span>`;
  }
  updateVoiceCallStatus(voiceCallMuted ? 'idle' : 'listening', voiceCallMuted ? 'Muted' : 'Listening...');
}

function endVoiceCall() {
  teardownVoiceCallMode();
  setModeSwitchUI('chat');
  setSystemState('idle');
  addMessage('system', 'Voice call ended.');
}

function teardownVoiceCallMode() {
  voiceCallActive = false;
  isCurrentlySpeaking = false;
  stopTtsPlayback();
  if (vadInstance) {
    try { vadInstance.pause(); vadInstance.destroy(); } catch (e) {}
    vadInstance = null;
  }
  const container = document.getElementById('voiceCallContainer');
  if (container) container.classList.remove('active');
}

// ============================================================================
// Video Call Mode (Webcam + VAD/PTT + VLM)
// ============================================================================
async function setupVideoCallMode() {
  log('Setting up Video Call mode...');
  const workspaceBody = document.getElementById('workspaceBody');
  const container = document.getElementById('videoCallContainer');
  if (workspaceBody) workspaceBody.classList.add('video-mode-active');
  if (container) container.classList.add('active');

  initAllWaveforms();
  startWaveformLoop();

  try {
    const video = document.getElementById('videoCallWebcam');
    videoCallStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    if (video) {
      video.srcObject = videoCallStream;
      await video.play();
    }
    const status = document.getElementById('videoCallWebcamStatus');
    if (status) status.textContent = 'Camera Active';
  } catch (err) {
    log(`Webcam notice: ${err.message}`);
    const status = document.getElementById('videoCallWebcamStatus');
    if (status) status.textContent = 'Camera Unavailable';
  }

  try {
    if (typeof vad === 'undefined') throw new Error('Local VAD bundle not loaded');
    videoCallVadInstance = await vad.MicVAD.new(createLocalVadConfig({
      onSpeechStart: () => {
        if (videoCallMuted || pttMode) return;
        if (videoCallProcessing || (isTtsPlaying && !bargeInEnabled)) return;
        if (isTtsPlaying && bargeInEnabled) stopTtsPlayback();
        currentUserMsg = null;
        videoCallSpeaking = true;
        updateVideoCallStatus('hearing', 'Listening...');
      },
      onSpeechEnd: (audio) => {
        if (videoCallMuted || pttMode || videoCallProcessing || isTtsPlaying) return;
        const energy = calculateRmsEnergy(audio);
        updateEnergyMeter(energy);
        const now = Date.now();
        if (lastTtsEndTime > 0 && now - lastTtsEndTime < POST_TTS_COOLDOWN_MS) return;
        if (now - lastVideoCallSendTime < VIDEO_CALL_DEBOUNCE_MS) return;
        if (energy < energyThreshold) {
          videoCallSpeaking = false;
          updateVideoCallStatus('listening', 'Listening...');
          return;
        }
        videoCallSpeaking = false;
        videoCallProcessing = true;
        updateVideoCallStatus('processing', 'Analyzing frame...');
        sendVideoCallData(audio);
      },
      onVADMisfire: () => {
        if (!videoCallSpeaking) updateVideoCallStatus('listening', 'Listening...');
      }
    }));

    if (!pttMode) {
      videoCallVadInstance.start();
      updateVideoCallStatus('listening', 'Listening...');
    } else {
      videoCallVadInstance.pause();
      updateVideoCallStatus('listening', 'Hold Space to talk');
    }
    videoCallActive = true;
  } catch (error) {
    log(`Video VAD notice: ${error.message}`);
    videoCallActive = true;
    updateVideoCallStatus('listening', 'Hold Space to talk');
  }
}

function updateVideoCallStatus(state, text) {
  const statusEl = document.getElementById('videoCallStatus');
  const textEl = document.getElementById('videoCallStatusText');
  if (statusEl) statusEl.className = `video-call-status voice-call-status ${state}`;
  if (textEl) textEl.textContent = text;

  if (state === 'listening' || state === 'hearing') {
    setSystemState('listening', text);
  } else if (state === 'processing') {
    setSystemState('thinking', text);
  } else if (state === 'speaking') {
    setSystemState('speaking', text);
  }
}

function captureWebcamBase64() {
  const video = document.getElementById('videoCallWebcam');
  if (!videoCallCameraOn || !video || !video.srcObject) return null;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.82).split(',')[1];
}

async function sendVideoCallData(audioFloat32) {
  const now = Date.now();
  if (now - lastVideoCallSendTime < VIDEO_CALL_DEBOUNCE_MS) return;
  lastVideoCallSendTime = now;

  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    updateVideoCallStatus('listening', 'Disconnected');
    return;
  }

  if (videoCallVadInstance && !pttMode) {
    try { videoCallVadInstance.pause(); } catch (e) {}
  }

  const imageBase64 = captureWebcamBase64();
  const wavBlob = float32ToWav(audioFloat32, 16000);
  const reader = new FileReader();
  reader.onload = () => {
    const audioBase64 = reader.result.split(',')[1];
    const payload = {
      type: 'video_call_data',
      audio: audioBase64,
      image: imageBase64,
      format: 'wav'
    };
    const mainPrompt = document.getElementById('systemPromptInput')?.value?.trim();
    if (mainPrompt) payload.system_prompt = mainPrompt;
    voiceWs.send(JSON.stringify(payload));
  };
  reader.readAsDataURL(wavBlob);
}

function toggleVideoCallMute() {
  videoCallMuted = !videoCallMuted;
  const btn = document.getElementById('videoCallMuteBtn');
  if (btn) {
    btn.classList.toggle('muted', videoCallMuted);
    btn.innerHTML = `${window.SparkIcons.iconSvg(videoCallMuted ? 'mic-off' : 'mic', 14)}<span>${videoCallMuted ? 'Unmute' : 'Mute'}</span>`;
  }
  updateVideoCallStatus('listening', videoCallMuted ? 'Muted' : (pttMode ? 'Hold Space to talk' : 'Listening...'));
}

function toggleVideoCallCamera() {
  videoCallCameraOn = !videoCallCameraOn;
  const btn = document.getElementById('videoCallCameraBtn');
  const status = document.getElementById('videoCallWebcamStatus');
  if (btn) {
    btn.classList.toggle('off', !videoCallCameraOn);
    btn.innerHTML = `${window.SparkIcons.iconSvg(videoCallCameraOn ? 'video' : 'video-off', 14)}<span>${videoCallCameraOn ? 'Camera On' : 'Camera Off'}</span>`;
  }
  if (status) status.textContent = videoCallCameraOn ? 'Camera Active' : 'Camera Off';
  if (videoCallStream) {
    videoCallStream.getVideoTracks().forEach((t) => { t.enabled = videoCallCameraOn; });
  }
}

function endVideoCall() {
  teardownVideoCallMode();
  setModeSwitchUI('chat');
  setSystemState('idle');
  addMessage('system', 'Video call ended.');
}

function teardownVideoCallMode() {
  videoCallActive = false;
  videoCallSpeaking = false;
  stopTtsPlayback();
  if (videoCallVadInstance) {
    try { videoCallVadInstance.pause(); videoCallVadInstance.destroy(); } catch (e) {}
    videoCallVadInstance = null;
  }
  if (videoCallStream) {
    videoCallStream.getTracks().forEach((t) => t.stop());
    videoCallStream = null;
  }
  cleanupPttAudio();
  const workspaceBody = document.getElementById('workspaceBody');
  const container = document.getElementById('videoCallContainer');
  if (workspaceBody) workspaceBody.classList.remove('video-mode-active');
  if (container) container.classList.remove('active');
  const video = document.getElementById('videoCallWebcam');
  if (video) video.srcObject = null;
}

// ============================================================================
// Input Mode (VAD vs PTT) & Push-to-Talk Recording
// ============================================================================
function setInputMode(mode) {
  pttMode = mode === 'ptt';
  const vadBtn = document.getElementById('vadModeBtn');
  const pttBtn = document.getElementById('pttModeBtn');
  const pttContainer = document.getElementById('videoCallPttContainer');

  if (vadBtn) vadBtn.classList.toggle('active', !pttMode);
  if (pttBtn) pttBtn.classList.toggle('active', pttMode);
  if (pttContainer) pttContainer.style.display = pttMode ? 'block' : 'none';

  if (pttMode) {
    if (videoCallVadInstance) {
      try { videoCallVadInstance.pause(); } catch (e) {}
    }
    initPttAudio();
    if (videoCallActive) updateVideoCallStatus('listening', 'Hold Space to talk');
  } else {
    if (videoCallVadInstance && videoCallActive) {
      try { videoCallVadInstance.start(); } catch (e) {}
    }
    cleanupPttAudio();
    if (videoCallActive) updateVideoCallStatus('listening', 'Listening...');
  }
}

async function initPttAudio() {
  try {
    if (pttStream) return;
    pttStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
    });
    pttAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  } catch (err) {
    log(`PTT audio init notice: ${err.message}`);
  }
}

function cleanupPttAudio() {
  if (pttMediaRecorder && pttMediaRecorder.state !== 'inactive') {
    try { pttMediaRecorder.stop(); } catch (e) {}
  }
  pttMediaRecorder = null;
  if (pttStream) {
    pttStream.getTracks().forEach((t) => t.stop());
    pttStream = null;
  }
  if (pttAudioContext) {
    try { pttAudioContext.close(); } catch (e) {}
    pttAudioContext = null;
  }
  pttAudioChunks = [];
  pttRecording = false;
  pttProcessing = false;
}

async function startPttRecording(event) {
  if (event && event.type === 'touchstart') event.preventDefault();
  if (!pttMode || pttRecording || videoCallMuted) return;

  pttRecording = true;
  pttAudioChunks = [];
  const pttBtn = document.getElementById('videoCallPttBtn');
  if (pttBtn) pttBtn.classList.add('recording');
  updateVideoCallStatus('hearing', 'Listening...');

  if (isTtsPlaying && bargeInEnabled) stopTtsPlayback();

  try {
    if (!pttStream) await initPttAudio();
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
    pttMediaRecorder = new MediaRecorder(pttStream, { mimeType });
    pttMediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) pttAudioChunks.push(e.data);
    };
    pttMediaRecorder.onstop = async () => {
      if (pttProcessing || pttAudioChunks.length === 0) return;
      pttProcessing = true;
      updateVideoCallStatus('processing', 'Analyzing...');
      const blob = new Blob(pttAudioChunks, { type: 'audio/webm' });
      pttAudioChunks = [];
      try {
        await processPttAudio(blob);
      } finally {
        pttProcessing = false;
      }
    };
    pttMediaRecorder.start(100);
  } catch (err) {
    log(`PTT recording error: ${err.message}`);
    pttRecording = false;
  }
}

function stopPttRecording(event) {
  if (event && event.type === 'touchend') event.preventDefault();
  if (!pttMode || !pttRecording) return;
  pttRecording = false;
  const pttBtn = document.getElementById('videoCallPttBtn');
  if (pttBtn) pttBtn.classList.remove('recording');
  if (pttMediaRecorder && pttMediaRecorder.state !== 'inactive') {
    pttMediaRecorder.stop();
  }
}

async function processPttAudio(audioBlob) {
  try {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    let audioFloat32 = channelData;
    if (audioBuffer.sampleRate !== 16000) {
      const ratio = 16000 / audioBuffer.sampleRate;
      const newLength = Math.round(channelData.length * ratio);
      audioFloat32 = new Float32Array(newLength);
      for (let i = 0; i < newLength; i++) {
        const srcIdx = i / ratio;
        const lo = Math.floor(srcIdx);
        const hi = Math.min(lo + 1, channelData.length - 1);
        const t = srcIdx - lo;
        audioFloat32[i] = channelData[lo] * (1 - t) + channelData[hi] * t;
      }
    }
    await ctx.close();
    sendVideoCallData(audioFloat32);
  } catch (err) {
    log(`PTT decode error: ${err.message}`);
  }
}

// Direct Mic Push-to-Talk in Bottom Control Bar
async function startRecording() {
  if (isRecording) return;
  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    log('Cannot record: WebSocket not connected');
    return;
  }

  const pttBtn = document.getElementById('pushToTalkBtn');
  if (pttBtn) pttBtn.classList.add('recording');
  setSystemState('listening', 'Listening');
  startWaveformLoop();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    log('Microphone API unavailable in this context');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 16000 }
    });
    const codecs = ['audio/webm;codecs=opus', 'audio/webm;codecs=vorbis', 'audio/webm'];
    const selectedMimeType = codecs.find((c) => MediaRecorder.isTypeSupported(c));
    mediaRecorder = new MediaRecorder(stream, selectedMimeType ? { mimeType: selectedMimeType } : {});

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0 && voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(e.data);
      }
    };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(JSON.stringify({ type: 'asr_end' }));
      }
    };

    currentUserMsg = null;
    isRecording = true;
    mediaRecorder.start(150);
    log('Push-to-talk recording started');
  } catch (err) {
    log(`Microphone access notice: ${err.message}`);
  }
}

function stopRecording() {
  const pttBtn = document.getElementById('pushToTalkBtn');
  if (pttBtn) pttBtn.classList.remove('recording');

  if (!isRecording) {
    if (currentSystemState === 'listening' && !voiceCallActive && !videoCallActive) {
      setSystemState('idle');
    }
    return;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  setSystemState('thinking');
  log('Push-to-talk recording stopped');
}

// ============================================================================
// WebAudio TTS Playback & Barge-In
// ============================================================================
function initializeAudioContext() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    nextPlayTime = audioContext.currentTime;
  }
  if (audioContext.state === 'suspended') {
    return audioContext.resume().catch(() => {});
  }
  return Promise.resolve();
}

function initializeAudioOnConnection() {
  initializeAudioContext().catch(() => {});
}

function stopTtsPlayback() {
  ttsAborted = true;
  if (masterGainNode && audioContext) {
    masterGainNode.gain.setValueAtTime(0, audioContext.currentTime);
  }
  if (activeAudioSources.length > 0) {
    activeAudioSources.forEach((src) => {
      try { src.stop(); src.disconnect(); } catch (e) {}
    });
    activeAudioSources = [];
  }
  nextPlayTime = null;
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'abort_tts' }));
  }
  isTtsPlaying = false;
  if (videoCallActive && videoCallVadInstance && !pttMode) {
    try { videoCallVadInstance.start(); } catch (e) {}
  }
}

async function handleAudioChunk(data) {
  if (ttsAborted) return;
  await initializeAudioContext();
  if (!audioContext) return;

  const arrayBuffer = data instanceof Blob ? await data.arrayBuffer() : data;
  if (arrayBuffer.byteLength === 0) return;
  const int16Data = new Int16Array(arrayBuffer);
  if (int16Data.length === 0) return;

  const float32Data = new Float32Array(int16Data.length);
  for (let i = 0; i < float32Data.length; i++) {
    float32Data[i] = int16Data[i] / 32768.0;
  }

  try {
    const buffer = audioContext.createBuffer(1, float32Data.length, 24000);
    buffer.copyToChannel(float32Data, 0);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;

    if (!masterGainNode) {
      masterGainNode = audioContext.createGain();
      masterGainNode.connect(audioContext.destination);
    }
    source.connect(masterGainNode);
    activeAudioSources.push(source);

    source.onended = () => {
      const idx = activeAudioSources.indexOf(source);
      if (idx > -1) activeAudioSources.splice(idx, 1);
      if (activeAudioSources.length === 0) {
        isTtsPlaying = false;
        videoCallProcessing = false;
        lastTtsEndTime = Date.now();
        if (voiceCallActive && !voiceCallMuted) {
          updateVoiceCallStatus('listening', 'Listening...');
        } else if (videoCallActive && !videoCallMuted) {
          updateVideoCallStatus('listening', pttMode ? 'Hold Space to talk' : 'Listening...');
        } else if (Object.values(hermesTasks).some((t) => t.status === 'running' || t.status === 'queued')) {
          setSystemState('agent_working');
        } else {
          setSystemState('idle');
        }
        if (videoCallActive && videoCallVadInstance && !pttMode) {
          setTimeout(() => {
            if (!isTtsPlaying && videoCallVadInstance) {
              try { videoCallVadInstance.start(); } catch (e) {}
            }
          }, 400);
        }
      }
    };

    if (nextPlayTime === null || nextPlayTime < audioContext.currentTime) {
      nextPlayTime = audioContext.currentTime;
    }
    source.start(nextPlayTime);
    nextPlayTime += buffer.duration;
  } catch (e) {
    log(`Audio playback error: ${e.message}`);
  }
}

// ============================================================================
// Face Recognition Management (/api/face/*)
// ============================================================================
async function refreshEnrolledFaces() {
  const container = document.getElementById('enrolledFacesList');
  if (!container) return;
  try {
    const res = await fetch('/api/face/list');
    const data = await res.json();
    const faces = data.faces || [];
    if (faces.length === 0) {
      container.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">No enrolled faces yet.</span>';
      return;
    }
    container.innerHTML = faces.map((name) => `
      <span class="artifact-chip">
        <span>${escapeHtml(name)}</span>
        <button type="button" onclick="deleteEnrolledFace('${escapeHtml(name)}')" title="Remove face" aria-label="Remove ${escapeHtml(name)}">
          ${window.SparkIcons.iconSvg('x', 12)}
        </button>
      </span>
    `).join('');
  } catch (e) {
    container.innerHTML = '<span style="font-size:12px;color:var(--text-muted);">Face service offline</span>';
  }
}

async function enrollCurrentFace() {
  const nameInput = document.getElementById('enrollFaceNameInput');
  const name = nameInput?.value?.trim();
  if (!name) return;
  const img = captureWebcamBase64();
  if (!img) {
    addMessage('system', 'Start Video Call mode first to capture a camera frame for face enrollment.');
    return;
  }
  try {
    const res = await fetch('/api/face/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: img })
    });
    const data = await res.json();
    if (data.success) {
      if (nameInput) nameInput.value = '';
      await refreshEnrolledFaces();
    } else {
      log(`Face enroll error: ${data.error}`);
    }
  } catch (e) {
    log(`Face enroll request failed: ${e.message}`);
  }
}

async function deleteEnrolledFace(name) {
  try {
    await fetch(`/api/face/${encodeURIComponent(name)}`, { method: 'DELETE' });
    await refreshEnrolledFaces();
  } catch (e) {
    log(`Face delete error: ${e.message}`);
  }
}
