/* ==========================================================================
   Spark Realtime — WebSocket Client, Latency Telemetry & Message Router
   ========================================================================== */

let pingIntervalId = null;

function setConnectionStatus(status) {
  const connectionStatusEl = document.getElementById('connectionStatus');
  const disconnectBtn = document.getElementById('disconnectBtn');
  const systemPromptInput = document.getElementById('systemPromptInput');
  const pushToTalkBtn = document.getElementById('pushToTalkBtn');

  if (connectionStatusEl) {
    connectionStatusEl.className = `connection-status ${status}`;
    const dot = connectionStatusEl.querySelector('.status-dot');
    if (dot) dot.className = `status-dot ${status}`;
    const label = connectionStatusEl.querySelector('.connection-status-text');
    const statusText = {
      connected: 'Connected',
      disconnected: 'Disconnected',
      connecting: 'Connecting...'
    };
    if (label) label.textContent = statusText[status] || status;
  }

  if (disconnectBtn) {
    if (status === 'connected') {
      disconnectBtn.textContent = 'Disconnect';
      disconnectBtn.disabled = false;
      disconnectBtn.classList.remove('connect-state');
    } else if (status === 'connecting') {
      disconnectBtn.textContent = 'Connecting...';
      disconnectBtn.disabled = true;
    } else {
      disconnectBtn.textContent = 'Connect';
      disconnectBtn.disabled = false;
      disconnectBtn.classList.add('connect-state');
      updateLatencyBadge(null);
    }
  }

  if (systemPromptInput) {
    systemPromptInput.disabled = status === 'connected' || status === 'connecting';
  }
  if (pushToTalkBtn) {
    pushToTalkBtn.disabled = status !== 'connected';
  }

  updateTextInputState();
}

function updateTextInputState() {
  const textInput = document.getElementById('textInput');
  const sendTextBtn = document.getElementById('sendTextBtn');
  const isConnected = Boolean(voiceWs && voiceWs.readyState === WebSocket.OPEN);
  if (textInput) textInput.disabled = !isConnected;
  if (sendTextBtn) sendTextBtn.disabled = !isConnected;
}

function sendPingProbe() {
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
  }
}

function connectVoiceWebSocket() {
  if (voiceWs && (voiceWs.readyState === WebSocket.OPEN || voiceWs.readyState === WebSocket.CONNECTING)) {
    return;
  }

  if (voiceWs) {
    try { voiceWs.close(); } catch (e) {}
    voiceWs = null;
  }

  const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
  const wsUrl = `${wsProtocol}${window.location.host}/ws/voice`;
  log(`Connecting to WebSocket: ${wsUrl}`);
  setConnectionStatus('connecting');
  voiceWs = new WebSocket(wsUrl);

  const connectionTimeout = setTimeout(() => {
    if (voiceWs && voiceWs.readyState !== WebSocket.OPEN) {
      log('Connection timeout');
      setConnectionStatus('disconnected');
      try { voiceWs.close(); } catch (e) {}
    }
  }, 10000);

  voiceWs.onopen = () => {
    clearTimeout(connectionTimeout);
    log('Voice WebSocket connected');
    setConnectionStatus('connected');
    if (currentSystemState === 'error') setSystemState('idle');

    sendPingProbe();
    if (pingIntervalId) clearInterval(pingIntervalId);
    pingIntervalId = setInterval(sendPingProbe, 10000);

    setTimeout(() => {
      if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
      if (pendingSystemPrompt) {
        voiceWs.send(JSON.stringify({ type: 'set_system_prompt', prompt: pendingSystemPrompt }));
        pendingSystemPrompt = null;
      } else {
        voiceWs.send(JSON.stringify({ type: 'get_system_prompt' }));
      }

      const voiceSelect = document.getElementById('voiceSelect');
      if (voiceSelect && voiceSelect.value) {
        voiceWs.send(JSON.stringify({ type: 'set_voice', voice: voiceSelect.value }));
      }

      syncToolsToServer();
    }, 80);
  };

  voiceWs.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      await handleAudioChunk(event.data);
      return;
    }
    if (typeof event.data === 'string') {
      try {
        const data = JSON.parse(event.data);
        await handleMessage(data);
      } catch (e) {
        log(`Error parsing WS message: ${e.message}`);
      }
    }
  };

  voiceWs.onerror = () => {
    clearTimeout(connectionTimeout);
    log('WebSocket connection error');
    setConnectionStatus('disconnected');
  };

  voiceWs.onclose = (event) => {
    clearTimeout(connectionTimeout);
    if (pingIntervalId) {
      clearInterval(pingIntervalId);
      pingIntervalId = null;
    }
    setConnectionStatus('disconnected');
    voiceWs = null;

    if (event.code !== 1000 && !isManualDisconnect) {
      setTimeout(() => {
        if (!voiceWs || voiceWs.readyState === WebSocket.CLOSED) {
          connectVoiceWebSocket();
        }
      }, 2000);
    }
    isManualDisconnect = false;
  };
}

function syncToolsToServer() {
  const checkboxes = document.querySelectorAll('input[data-tool-checkbox]');
  const enabledTools = [];
  checkboxes.forEach((cb) => {
    if (cb.checked) enabledTools.push(cb.value);
  });
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'set_tools', tools: enabledTools }));
    log(`Tools synced: ${enabledTools.join(', ') || 'none'}`);
  }
}

async function handleMessage(data) {
  switch (data.type) {
    case 'connected':
      initializeAudioOnConnection();
      if (data.hermes) hermesStatus = data.hermes;
      if (Array.isArray(data.tasks)) {
        data.tasks.forEach((t) => { hermesTasks[t.task_id] = t; });
      }
      if (Array.isArray(data.artifacts)) {
        workspaceArtifacts = data.artifacts;
      }
      renderHermesPanel();
      sendPingProbe();
      break;

    case 'pong':
      if (data.ts) {
        updateLatencyBadge(Math.max(1, Date.now() - Number(data.ts)));
      }
      break;

    case 'asr_partial':
      removeEmptyState();
      setSystemState('listening', 'Transcribing');
      if (!currentUserMsg) {
        const msg = createMessageElement('user', '');
        getActiveConversationEl().appendChild(msg.container);
        currentUserMsg = msg;
      }
      currentUserMsg.content.textContent = data.text;
      scrollToBottom();
      break;

    case 'asr_final':
      removeEmptyState();
      if (!currentUserMsg) {
        if (!isRecording) {
          const msg = createMessageElement('user', data.text);
          getActiveConversationEl().appendChild(msg.container);
        }
      } else {
        currentUserMsg.content.textContent = data.text;
      }
      currentUserMsg = null;
      showThinkingIndicator();
      if (voiceCallActive) updateVoiceCallStatus('processing', 'Thinking...');
      if (videoCallActive) updateVideoCallStatus('processing', 'Analyzing...');
      break;

    case 'asr_result': {
      removeEmptyState();
      const asrText = data.text || data.content || '';
      if (asrText.trim()) {
        if (currentUserMsg) {
          currentUserMsg.content.textContent = asrText;
        } else {
          const userMsg = createMessageElement('user', asrText);
          getActiveConversationEl().appendChild(userMsg.container);
        }
        scrollToBottom();
        if (typeof saveCurrentChat === 'function') saveCurrentChat();
        showThinkingIndicator();
      }
      currentUserMsg = null;
      break;
    }

    case 'transient_response':
      hideThinkingIndicator();
      removeEmptyState();
      if (currentTransientMsg) {
        currentTransientMsg.container.remove();
      }
      currentTransientMsg = createMessageElement('assistant', data.text, true);
      getActiveConversationEl().appendChild(currentTransientMsg.container);
      scrollToBottom();
      break;

    case 'tool_invocation':
      setSystemState('agent_working', data.message || 'Running tool');
      break;

    case 'agent_started':
      hideThinkingIndicator();
      setSystemState('agent_working', `${data.agent_type || 'Agent'}`);
      if (data.agent_type === 'markdown_assistant') {
        openMarkdownEditor(data.task);
      } else if (data.agent_type === 'html_assistant') {
        openHtmlEditor(data.task);
      }
      break;

    case 'agent_markdown_chunk':
      if (markdownModalOpen) {
        const mdEditor = document.getElementById('agentMarkdownEditor');
        const mdPreview = document.getElementById('agentMarkdownPreview');
        if (data.done) {
          const statusEl = document.getElementById('markdownStatus');
          if (statusEl) statusEl.textContent = 'Document complete';
          setSystemState('idle');
        } else {
          markdownContent += data.content || '';
          if (mdEditor) {
            mdEditor.textContent = markdownContent;
            mdEditor.scrollTop = mdEditor.scrollHeight;
          }
          if (mdPreview) {
            mdPreview.innerHTML = renderMarkdownPreview(markdownContent, 'agentMarkdownPreview');
          }
        }
      }
      break;

    case 'agent_markdown_complete':
      removeEmptyState();
      setSystemState('idle');
      if (data.markdown) {
        const mdMsg = createMarkdownMessageElement(data.task, data.markdown);
        getActiveConversationEl().appendChild(mdMsg.container);
        scrollToBottom();
        if (typeof saveCurrentChat === 'function') saveCurrentChat();
      }
      break;

    case 'agent_html_chunk':
      if (htmlModalOpen) {
        const htmlEditor = document.getElementById('agentHtmlEditor');
        const htmlPreview = document.getElementById('agentHtmlPreview');
        const htmlCharCount = document.getElementById('htmlCharCount');
        if (data.done) {
          const statusEl = document.getElementById('htmlStatus');
          if (statusEl) statusEl.textContent = 'Page complete';
          if (htmlPreview && htmlContent) htmlPreview.srcdoc = htmlContent;
          setSystemState('idle');
        } else {
          const chunk = data.content || '';
          htmlContent += chunk;
          if (htmlEditor) {
            htmlEditor.textContent = htmlContent;
            htmlEditor.scrollTop = htmlEditor.scrollHeight;
          }
          if (htmlCharCount) htmlCharCount.textContent = `${htmlContent.length} chars`;
          if (htmlPreview && (htmlContent.includes('</body>') || htmlContent.length % 500 < chunk.length)) {
            htmlPreview.srcdoc = htmlContent;
          }
        }
      }
      break;

    case 'agent_html_complete':
      setSystemState('idle');
      if (htmlModalOpen && data.html) {
        htmlContent = data.html;
        const htmlPreview = document.getElementById('agentHtmlPreview');
        if (htmlPreview) htmlPreview.srcdoc = data.html;
      }
      break;

    case 'reasoning_started':
      hideThinkingIndicator();
      removeEmptyState();
      setSystemState('agent_working', 'Nemotron Reasoning');
      createReasoningMessage(data.problem);
      break;

    case 'reasoning_thinking':
      if (data.content) appendInlineThinking(data.content);
      break;

    case 'reasoning_content':
      if (data.content) appendInlineConclusion(data.content);
      break;

    case 'reasoning_complete':
      finalizeReasoningMessage(data.conclusion || data.thinking);
      if (!isTtsPlaying) setSystemState('idle');
      scrollToBottom();
      if (typeof saveCurrentChat === 'function') saveCurrentChat();
      break;

    case 'hermes_event':
      handleHermesEvent(data);
      break;

    case 'hermes_summary':
      handleHermesSummary(data);
      break;

    case 'comfy_progress':
      if (typeof handleComfyProgress === 'function') handleComfyProgress(data);
      break;

    case 'media_ready':
      if (typeof handleMediaReady === 'function') handleMediaReady(data);
      if (data.path) {
        switchAgentTab('artifacts');
        renderHermesPanel();
      }
      break;

    case 'workspace_updated':
      if (Array.isArray(data.artifacts)) {
        workspaceArtifacts = data.artifacts;
        renderHermesPanel();
      }
      break;

    case 'artifact_ready':
      if (data.path) {
        log(`Artifact ready: ${data.path}`);
        switchAgentTab('artifacts');
        renderHermesPanel();
      }
      break;

    case 'path_opened':
      log(`Host opened: ${data.opened || data.path || ''}`);
      break;

    case 'final_response':
    case 'llm_final': {
      hideThinkingIndicator();
      removeEmptyState();
      if (currentTransientMsg) {
        currentTransientMsg.container.remove();
        currentTransientMsg = null;
      }
      const text = data.text || data.content || '';
      if (text.trim()) {
        const finalMsg = createMessageElement('assistant', text);
        getActiveConversationEl().appendChild(finalMsg.container);
        scrollToBottom();
        if (typeof saveCurrentChat === 'function') saveCurrentChat();
      }
      if (!isTtsPlaying && currentSystemState === 'thinking') {
        setSystemState('idle');
      }
      break;
    }

    case 'tts_start':
      isTtsPlaying = true;
      ttsAborted = false;
      setSystemState('speaking', 'Speaking');
      if (videoCallActive && videoCallVadInstance && !pttMode) {
        try { videoCallVadInstance.pause(); } catch (e) {}
      }
      if (masterGainNode && audioContext) {
        masterGainNode.gain.setValueAtTime(1, audioContext.currentTime);
      }
      if (voiceCallActive) updateVoiceCallStatus('speaking', 'Speaking...');
      if (videoCallActive) updateVideoCallStatus('speaking', 'Speaking...');
      initializeAudioContext().then(() => {
        if (audioContext) nextPlayTime = audioContext.currentTime;
      });
      break;

    case 'tts_done':
      if (activeAudioSources.length === 0) {
        isTtsPlaying = false;
        videoCallProcessing = false;
        lastTtsEndTime = Date.now();
        if (voiceCallActive && !voiceCallMuted) {
          updateVoiceCallStatus('listening', 'Listening...');
        } else if (videoCallActive && !videoCallMuted) {
          updateVideoCallStatus('listening', pttMode ? 'Hold Space to talk' : 'Listening...');
        } else if (Object.values(hermesTasks).some((t) => t.status === 'running' || t.status === 'queued')) {
          setSystemState('agent_working', 'Agent Working');
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
      break;

    case 'system_prompt': {
      const promptInput = document.getElementById('systemPromptInput');
      if (promptInput && data.prompt) promptInput.value = data.prompt;
      break;
    }

    case 'voice_changed':
    case 'system_prompt_changed':
    case 'tools_changed':
    case 'reset_ack':
    case 'disconnect_ack':
      break;

    case 'error':
      hideThinkingIndicator();
      setSystemState('error', 'Error');
      log(`Server error: ${data.error}`);
      addMessage('system', `Error: ${data.error}`);
      videoCallProcessing = false;
      break;
  }
}
