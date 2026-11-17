/* ==========================================================================
   Spark Realtime — Main UI Controller, Sessions, Mode Switch & Shortcuts
   ========================================================================== */

function generateChatId() {
  return 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2, 9);
}

function saveChatsToStorage() {
  try {
    localStorage.setItem('spark_realtime_chats', JSON.stringify(chats));
  } catch (e) {
    console.error('Failed to save chats:', e);
  }
}

function loadChatsFromStorage() {
  try {
    const stored = localStorage.getItem('spark_realtime_chats');
    if (stored) {
      chats = JSON.parse(stored) || {};
      Object.values(chats).forEach((chat) => {
        if (Array.isArray(chat.messages) && chat.messages.length > 0) {
          chat.greeted = true;
        }
      });
    }
  } catch (e) {
    chats = {};
  }
}

function renderEmptyState(mode = 'chat') {
  const modeMeta = {
    chat: {
      icon: 'message-square',
      title: 'Ready on DGX Spark',
      hint: 'Hold Space or the mic to speak. Ask about Downloads or files on this Spark, or hand off heavy work to Hermes in the background.'
    },
    call: {
      icon: 'phone',
      title: 'Voice Call Active',
      hint: 'Silero VAD is listening for speech with automatic turn detection and low-latency Kokoro TTS playback.'
    },
    video: {
      icon: 'video',
      title: 'Video Call Active',
      hint: 'Camera frames are paired with your spoken questions for real-time Qwen3-VL scene analysis and face recognition.'
    }
  };
  const m = modeMeta[mode] || modeMeta.chat;
  return `
    <div class="empty-state">
      <div class="empty-state-badge">${window.SparkIcons.iconSvg(m.icon, 20)}</div>
      <div class="empty-state-text">${escapeHtml(m.title)}</div>
      <div class="empty-state-hint">${escapeHtml(m.hint)}</div>
      <div class="empty-state-shortcuts">
        <button type="button" class="shortcut-chip" onclick="switchWorkspaceMode('call')">
          ${window.SparkIcons.iconSvg('phone', 13)}
          <span>Voice Call</span>
        </button>
        <button type="button" class="shortcut-chip" onclick="switchWorkspaceMode('video')">
          ${window.SparkIcons.iconSvg('video', 13)}
          <span>Video Call</span>
        </button>
        <button type="button" class="shortcut-chip" onclick="focusHermesDelegateInput()">
          ${window.SparkIcons.iconSvg('terminal', 13)}
          <span>Delegate to Hermes</span>
        </button>
      </div>
    </div>
  `;
}

function focusHermesDelegateInput() {
  const appShell = document.getElementById('appShell');
  const agentPanel = document.getElementById('agentPanel');
  if (appShell) appShell.classList.remove('right-panel-collapsed');
  if (agentPanel && window.innerWidth <= 960) {
    agentPanel.classList.add('open');
    document.getElementById('sidebarOverlay')?.classList.add('active');
  }
  switchAgentTab('activity');
  document.getElementById('hermesDelegateInput')?.focus();
}

function renderChatList() {
  const chatListEl = document.getElementById('chatList');
  if (!chatListEl) return;

  const sortedChats = Object.values(chats).sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
  chatListEl.innerHTML = '';

  sortedChats.forEach((chat) => {
    const item = document.createElement('div');
    item.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
    item.onclick = () => {
      loadChat(chat.id);
      if (window.innerWidth <= 960) closeMobileDrawers();
    };

    const modeIcon = chat.mode === 'video' ? 'video' : (chat.mode === 'call' ? 'phone' : 'message-square');
    const cleanTitle = (chat.title || 'New Session').replace(/^[^\w\s]+\s*/, '') || 'Session';
    const time = chat.timestamp ? new Date(chat.timestamp) : new Date();
    const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    item.innerHTML = `
      <div class="chat-item-content">
        <div class="chat-item-header">
          <span class="chat-item-mode-icon">${window.SparkIcons.iconSvg(modeIcon, 13)}</span>
          <div class="chat-item-title">${escapeHtml(cleanTitle)}</div>
        </div>
        <div class="chat-item-preview">${escapeHtml(chat.preview || 'No messages yet')}</div>
        <div class="chat-item-time">${escapeHtml(timeStr)}</div>
      </div>
      <button type="button" class="chat-item-delete" onclick="deleteChat('${escapeHtml(chat.id)}', event)" title="Delete session" aria-label="Delete session">
        ${window.SparkIcons.iconSvg('trash-2', 13)}
      </button>
    `;
    chatListEl.appendChild(item);
  });
}

function saveCurrentChat() {
  if (!currentChatId || !chats[currentChatId]) return;
  const activeEl = getActiveConversationEl();
  if (!activeEl) return;

  const messages = [];
  activeEl.querySelectorAll('.message').forEach((msgEl) => {
    if (msgEl.classList.contains('message-system') || msgEl.classList.contains('message-transient')) return;
    const role = msgEl.classList.contains('message-user') ? 'user' : 'assistant';
    const contentEl = msgEl.querySelector('.message-content');
    const text = contentEl?.textContent?.trim();
    if (text) messages.push({ role, content: text });
  });

  const chat = chats[currentChatId];
  const prevCount = chat.messages?.length || 0;
  chat.messages = messages;

  const firstUser = messages.find((m) => m.role === 'user');
  if (firstUser) {
    chat.title = firstUser.content.slice(0, 46) + (firstUser.content.length > 46 ? '...' : '');
    chat.preview = firstUser.content.slice(0, 80);
  }
  const lastMsg = messages[messages.length - 1];
  if (lastMsg) {
    chat.preview = lastMsg.content.slice(0, 80);
  }
  if (messages.length > prevCount) {
    chat.timestamp = new Date().toISOString();
  }

  const sessionTitleEl = document.getElementById('activeSessionTitle');
  if (sessionTitleEl) {
    sessionTitleEl.textContent = (chat.title || 'New Session').replace(/^[^\w\s]+\s*/, '');
  }

  saveChatsToStorage();
  renderChatList();
}

function loadChat(chatId) {
  if (currentChatId && chats[currentChatId] && currentChatId !== chatId) {
    saveCurrentChat();
  }

  teardownVoiceCallMode();
  teardownVideoCallMode();

  const chatForMode = chats[chatId];
  const mode = (chatForMode && chatForMode.mode) || 'chat';
  setModeSwitchUI(mode);

  currentChatId = chatId;
  const chat = chats[chatId];
  renderChatList();

  const conversationEl = getActiveConversationEl();
  if (!conversationEl) return;
  currentTransientMsg = null;

  const sessionTitleEl = document.getElementById('activeSessionTitle');
  if (sessionTitleEl) {
    sessionTitleEl.textContent = ((chat && chat.title) || 'New Session').replace(/^[^\w\s]+\s*/, '');
  }

  if (chat && Array.isArray(chat.messages) && chat.messages.length > 0) {
    conversationEl.innerHTML = '';
    chat.messages.forEach((m) => {
      const el = createMessageElement(m.role, m.content);
      conversationEl.appendChild(el.container);
    });
    scrollToBottom();
  } else {
    conversationEl.innerHTML = renderEmptyState(chat?.mode || 'chat');
  }

  if (mode === 'call' || mode === 'video') {
    pendingHandsFreeSetup = true;
  }
}

function deleteChat(chatId, event) {
  if (event) event.stopPropagation();
  if (!chats[chatId]) return;
  delete chats[chatId];
  saveChatsToStorage();

  if (currentChatId === chatId) {
    const remaining = Object.keys(chats);
    if (remaining.length > 0) {
      loadChat(remaining[0]);
    } else {
      createDefaultSession();
    }
  }
  renderChatList();
}

function clearPreviousChats() {
  const otherIds = Object.keys(chats).filter((id) => id !== currentChatId);
  otherIds.forEach((id) => delete chats[id]);
  saveChatsToStorage();
  renderChatList();
}

function clearChat() {
  const conversationEl = getActiveConversationEl();
  if (conversationEl) {
    conversationEl.innerHTML = renderEmptyState(selectedChatMode);
  }
  if (currentChatId && chats[currentChatId]) {
    chats[currentChatId].messages = [];
    chats[currentChatId].preview = '';
    chats[currentChatId].greeted = false;
    saveChatsToStorage();
    renderChatList();
  }
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'reset' }));
  }
}

function createDefaultSession(mode = 'call') {
  const chatId = generateChatId();
  const titles = { chat: 'Voice & Text Session', call: 'Voice Call', video: 'Video Call' };
  chats[chatId] = {
    id: chatId,
    title: titles[mode] || 'New Session',
    preview: '',
    timestamp: new Date().toISOString(),
    messages: [],
    mode,
    greeted: false
  };
  currentChatId = chatId;
  saveChatsToStorage();
  renderChatList();
  loadChat(chatId);
}

// Mode Switching & New Session Modal
function setModeSwitchUI(mode) {
  selectedChatMode = mode;
  document.querySelectorAll('.mode-seg-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  const camToggleBtn = document.getElementById('bottomCameraToggleBtn');
  if (camToggleBtn) {
    camToggleBtn.classList.toggle('active', mode === 'video');
  }
}

function switchWorkspaceMode(mode) {
  teardownVoiceCallMode();
  teardownVideoCallMode();
  setModeSwitchUI(mode);

  if (currentChatId && chats[currentChatId]) {
    chats[currentChatId].mode = mode;
    saveChatsToStorage();
    renderChatList();
  }

  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    connectVoiceWebSocket();
  }

  if (mode === 'call') {
    pendingHandsFreeSetup = true;
    setupVoiceCallMode();
  } else if (mode === 'video') {
    pendingHandsFreeSetup = true;
    setupVideoCallMode();
  } else {
    setSystemState('idle');
  }
}

function openChatModeModal() {
  const modal = document.getElementById('chatModeModal');
  if (modal) {
    modal.classList.add('active');
    modal.style.display = 'flex';
  }
  selectChatMode(selectedChatMode || 'call');
}

function closeChatModeModal() {
  const modal = document.getElementById('chatModeModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
}

function createNewChat() {
  openChatModeModal();
}

function selectChatMode(mode) {
  selectedChatMode = mode;
  document.querySelectorAll('.chat-mode-card').forEach((card) => {
    card.classList.toggle('selected', card.dataset.mode === mode);
  });
}

function startSelectedChat() {
  closeChatModeModal();
  closeMobileDrawers();
  if (currentChatId && chats[currentChatId]) saveCurrentChat();
  createDefaultSession(selectedChatMode);
  resetServerConversation();
  switchWorkspaceMode(selectedChatMode);
  setTimeout(() => requestSessionGreetingIfNeeded(), 120);
}

// Settings & Prompts Modal
function openSettingsModal() {
  refreshEnrolledFaces();
  const modal = document.getElementById('settingsModal');
  if (modal) {
    modal.classList.add('active');
    modal.style.display = 'flex';
  }
}

function closeSettingsModal() {
  const modal = document.getElementById('settingsModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
}

function toggleConfig() {
  openSettingsModal();
}

function applyVisionTemplate(templateKey) {
  if (!templateKey || !visionTemplates[templateKey]) return;
  const promptInput = document.getElementById('systemPromptInput');
  if (promptInput) {
    promptInput.value = visionTemplates[templateKey];
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
      voiceWs.send(JSON.stringify({ type: 'set_system_prompt', prompt: promptInput.value.trim() }));
    } else {
      pendingSystemPrompt = promptInput.value.trim();
    }
  }
}

function saveSystemPromptNow() {
  const promptInput = document.getElementById('systemPromptInput');
  const prompt = promptInput?.value?.trim();
  if (!prompt) return;
  pendingSystemPrompt = prompt;
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'set_system_prompt', prompt }));
  }
}

// Sidebar / Panel Toggles
function toggleLeftSidebar() {
  if (window.innerWidth <= 960) {
    const sidebar = document.getElementById('chatSidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const isOpen = sidebar?.classList.toggle('open');
    document.getElementById('agentPanel')?.classList.remove('open');
    if (overlay) overlay.classList.toggle('active', Boolean(isOpen));
  } else {
    document.getElementById('appShell')?.classList.toggle('left-panel-collapsed');
  }
}

function toggleRightPanel() {
  if (window.innerWidth <= 960) {
    const panel = document.getElementById('agentPanel');
    const overlay = document.getElementById('sidebarOverlay');
    const isOpen = panel?.classList.toggle('open');
    document.getElementById('chatSidebar')?.classList.remove('open');
    if (overlay) overlay.classList.toggle('active', Boolean(isOpen));
  } else {
    document.getElementById('appShell')?.classList.toggle('right-panel-collapsed');
  }
}

function closeMobileDrawers() {
  document.getElementById('chatSidebar')?.classList.remove('open');
  document.getElementById('agentPanel')?.classList.remove('open');
  document.getElementById('sidebarOverlay')?.classList.remove('active');
}

function toggleMobileSidebar() {
  toggleLeftSidebar();
}
function closeMobileSidebar() {
  closeMobileDrawers();
}

// Text Message Sending
function sendTextMessage() {
  const textInput = document.getElementById('textInput');
  if (!textInput || !textInput.value.trim()) return;
  const messageText = textInput.value.trim();

  if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
    connectVoiceWebSocket();
    return;
  }

  textInput.value = '';
  textInput.style.height = 'auto';
  removeEmptyState();

  const userMsg = createMessageElement('user', messageText);
  getActiveConversationEl().appendChild(userMsg.container);
  scrollToBottom();
  saveCurrentChat();

  voiceWs.send(JSON.stringify({ type: 'text_message', text: messageText }));
  showThinkingIndicator();
}

// Initialization & Keyboard Shortcuts
async function loadDefaultSystemPrompt() {
  try {
    const res = await fetch('/api/default_prompt');
    if (!res.ok) return;
    const data = await res.json();
    const promptInput = document.getElementById('systemPromptInput');
    if (data.prompt && promptInput && !promptInput.value.trim()) {
      promptInput.value = data.prompt;
      pendingSystemPrompt = data.prompt;
    }
    if (data.templates) {
      visionTemplates = data.templates;
    }
  } catch (err) {
    log(`Default prompt load notice: ${err.message}`);
  }
}

function setupKeyboardShortcuts() {
  let spacePttActive = false;

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeChatModeModal();
      closeSettingsModal();
      if (markdownModalOpen) closeMarkdownEditor();
      if (htmlModalOpen) closeHtmlEditor();
      return;
    }

    const active = document.activeElement;
    const isTyping = active && (
      active.tagName === 'INPUT' ||
      active.tagName === 'TEXTAREA' ||
      active.tagName === 'SELECT' ||
      active.isContentEditable
    );
    if (isTyping) return;

    // Spacebar or Digit0 for Push-to-Talk
    if ((e.code === 'Space' || e.code === 'Digit0') && !e.repeat) {
      e.preventDefault();
      if (videoCallActive && pttMode) {
        startPttRecording();
      } else if (!spacePttActive) {
        spacePttActive = true;
        startRecording();
      }
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' || e.code === 'Digit0') {
      if (videoCallActive && pttMode && pttRecording) {
        e.preventDefault();
        stopPttRecording();
      } else if (spacePttActive) {
        e.preventDefault();
        spacePttActive = false;
        stopRecording();
      }
    }
  });

  window.addEventListener('blur', () => {
    if (spacePttActive) {
      spacePttActive = false;
      stopRecording();
    }
    if (pttRecording) stopPttRecording();
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  if (window.SparkIcons) window.SparkIcons.hydrateIcons();
  initAllWaveforms();
  startWaveformLoop();
  setupKeyboardShortcuts();

  loadChatsFromStorage();
  const existingIds = Object.keys(chats);
  if (existingIds.length === 0) {
    createDefaultSession('call');
  } else {
    const newest = existingIds.sort((a, b) => new Date(chats[b].timestamp || 0) - new Date(chats[a].timestamp || 0))[0];
    loadChat(newest);
  }

  // Bind controls
  const clearBtn = document.getElementById('clearBtn');
  if (clearBtn) clearBtn.onclick = clearChat;

  const disconnectBtn = document.getElementById('disconnectBtn');
  if (disconnectBtn) {
    disconnectBtn.onclick = () => {
      if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        isManualDisconnect = true;
        stopTtsPlayback();
        voiceWs.send(JSON.stringify({ type: 'disconnect' }));
        voiceWs.close();
      } else {
        connectVoiceWebSocket();
      }
    };
  }

  const voiceSelect = document.getElementById('voiceSelect');
  if (voiceSelect) {
    voiceSelect.onchange = (e) => {
      if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
        voiceWs.send(JSON.stringify({ type: 'set_voice', voice: e.target.value }));
      }
    };
  }

  document.querySelectorAll('input[data-tool-checkbox]').forEach((cb) => {
    cb.addEventListener('change', syncToolsToServer);
  });

  const textInput = document.getElementById('textInput');
  if (textInput) {
    textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendTextMessage();
      }
    });
    textInput.addEventListener('input', () => {
      textInput.style.height = 'auto';
      textInput.style.height = Math.min(textInput.scrollHeight, 110) + 'px';
    });
  }

  const hermesInput = document.getElementById('hermesDelegateInput');
  if (hermesInput) {
    hermesInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submitHermesTaskFromUI();
      }
    });
  }

  const pttBtn = document.getElementById('pushToTalkBtn');
  if (pttBtn) {
    pttBtn.addEventListener('mousedown', (e) => { e.preventDefault(); startRecording(); });
    pttBtn.addEventListener('mouseup', (e) => { e.preventDefault(); stopRecording(); });
    pttBtn.addEventListener('mouseleave', () => { if (isRecording) stopRecording(); });
    pttBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); }, { passive: false });
    pttBtn.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); }, { passive: false });
  }

  initInputModeFromStorage();

  await loadDefaultSystemPrompt();
  await fetchHermesStatus();
  connectVoiceWebSocket();

  const bootMode = chats[currentChatId]?.mode;
  if (bootMode === 'call' || bootMode === 'video') {
    pendingHandsFreeSetup = true;
    setModeSwitchUI(bootMode);
  }
});
