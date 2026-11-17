/* ==========================================================================
   Spark Realtime — Hermes Agent Activity Panel, Workspace Artifacts & Editors
   ========================================================================== */

let markdownModalOpen = false;
let markdownContent = '';
let htmlModalOpen = false;
let htmlContent = '';
let currentReasoningContainer = null;
let currentThinkingEl = null;
let currentConclusionEl = null;
let inlineThinkingContent = '';
let inlineConclusionContent = '';

async function fetchHermesStatus() {
  try {
    const res = await fetch('/api/agents/status');
    if (!res.ok) return;
    const data = await res.json();
    if (data.hermes) {
      hermesStatus = data.hermes;
    }
    if (Array.isArray(data.tasks)) {
      data.tasks.forEach((t) => {
        hermesTasks[t.task_id] = t;
      });
    }
    if (Array.isArray(data.artifacts)) {
      workspaceArtifacts = data.artifacts;
    }
    if (data.comfyui && Array.isArray(data.comfyui.jobs)) {
      data.comfyui.jobs.forEach((j) => {
        if (j && j.job_id) comfyJobs[j.job_id] = j;
      });
    }
    renderHermesPanel();
  } catch (e) {
    log(`Hermes status poll notice: ${e.message}`);
  }
}

function switchAgentTab(tab) {
  activeAgentTab = tab;
  const tabActivity = document.getElementById('tabBtnActivity');
  const tabArtifacts = document.getElementById('tabBtnArtifacts');
  const viewActivity = document.getElementById('hermesActivityView');
  const viewArtifacts = document.getElementById('hermesArtifactsView');

  if (tabActivity) tabActivity.classList.toggle('active', tab === 'activity');
  if (tabArtifacts) tabArtifacts.classList.toggle('active', tab === 'artifacts');
  if (viewActivity) viewActivity.style.display = tab === 'activity' ? 'flex' : 'none';
  if (viewArtifacts) viewArtifacts.style.display = tab === 'artifacts' ? 'flex' : 'none';
}

function handleComfyProgress(payload) {
  if (!payload || !payload.job_id) return;
  comfyJobs[payload.job_id] = { ...comfyJobs[payload.job_id], ...payload };
  setSystemState('agent_working', 'ComfyUI');
  renderHermesPanel();
}

function handleMediaReady(payload) {
  if (!payload) return;
  if (payload.job_id && comfyJobs[payload.job_id]) {
    comfyJobs[payload.job_id] = { ...comfyJobs[payload.job_id], ...payload, status: payload.success === false ? 'failed' : 'completed' };
  }
  if (!isTtsPlaying) setSystemState('idle');
  renderHermesPanel();
}

function handleHermesEvent(payload) {
  if (!payload || !payload.task_id) return;
  const taskId = payload.task_id;
  const evtType = payload.event_type;

  if (!hermesTasks[taskId]) {
    hermesTasks[taskId] = payload.task || {
      task_id: taskId,
      task: payload.task?.task || 'Background task',
      category: payload.task?.category || 'general',
      status: 'running',
      created_at: payload.timestamp || Date.now() / 1000,
      steps: [],
      artifacts: [],
      output: '',
      thinking: ''
    };
  }

  const task = hermesTasks[taskId];
  if (payload.task) {
    Object.assign(task, payload.task);
  }

  if (evtType === 'task.queued') {
    task.status = 'queued';
    setSystemState('agent_working', 'Agent Queued');
  } else if (evtType === 'task.started') {
    task.status = 'running';
    setSystemState('agent_working', 'Agent Working');
  } else if (evtType === 'step.progress' || evtType === 'tool.started' || evtType === 'tool.completed' || evtType === 'tool.failed') {
    if (payload.step) {
      const exists = (task.steps || []).some((s) => s.id === payload.step.id);
      if (!exists) {
        task.steps = task.steps || [];
        task.steps.push(payload.step);
      }
    }
    setSystemState('agent_working', payload.tool ? `Tool: ${payload.tool}` : 'Agent Working');
  } else if (evtType === 'thinking.delta') {
    task.thinking = (task.thinking || '') + (payload.delta || '');
    setSystemState('agent_working', 'Agent Reasoning');
  } else if (evtType === 'message.delta') {
    if (!payload.task) {
      task.output = (task.output || '') + (payload.delta || '');
    }
    setSystemState('agent_working', 'Agent Streaming');
  } else if (evtType === 'artifact.created') {
    if (payload.artifact && !(task.artifacts || []).includes(payload.artifact)) {
      task.artifacts = task.artifacts || [];
      task.artifacts.push(payload.artifact);
    }
  } else if (evtType === 'task.completed') {
    task.status = 'completed';
    if (!isTtsPlaying) setSystemState('idle');
  } else if (evtType === 'task.failed') {
    task.status = 'failed';
    task.error = payload.error || task.error || 'Task failed';
    setSystemState('error', 'Agent Error');
  } else if (evtType === 'task.cancelled') {
    task.status = 'cancelled';
    setSystemState('idle');
  }

  renderHermesPanel();
}

function handleHermesSummary(data) {
  if (!data || !data.text) return;
  removeEmptyState();
  const msg = createMessageElement('assistant', data.text);
  getActiveConversationEl().appendChild(msg.container);
  scrollToBottom();
  if (typeof saveCurrentChat === 'function') saveCurrentChat();
}

function renderHermesPanel() {
  const badgeEl = document.getElementById('hermesStatusBadge');
  const countActivityEl = document.getElementById('activityCountBadge');
  const countArtifactsEl = document.getElementById('artifactsCountBadge');
  const tasksListEl = document.getElementById('hermesTasksList');
  const artifactsListEl = document.getElementById('workspaceArtifactsList');

  const taskList = Object.values(hermesTasks).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const comfyList = Object.values(comfyJobs).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const combinedActivity = [
    ...taskList.map((t) => ({ type: 'hermes', item: t, created_at: t.created_at || 0 })),
    ...comfyList.map((j) => ({ type: 'comfy', item: j, created_at: j.created_at || 0 })),
  ].sort((a, b) => b.created_at - a.created_at);
  const activeCount = taskList.filter((t) => t.status === 'running' || t.status === 'queued').length
    + comfyList.filter((j) => j.status === 'running' || j.status === 'queued').length;

  if (badgeEl) {
    if (activeCount > 0) {
      badgeEl.className = 'hermes-status-badge working';
      badgeEl.textContent = `${activeCount} running`;
    } else if (hermesStatus.configured || hermesStatus.available) {
      badgeEl.className = 'hermes-status-badge online';
      badgeEl.textContent = hermesStatus.transport && hermesStatus.transport !== 'unconfigured'
        ? hermesStatus.transport.replace('_', ' ')
        : 'ready';
    } else {
      badgeEl.className = 'hermes-status-badge';
      badgeEl.textContent = 'unconfigured';
    }
  }

  if (countActivityEl) countActivityEl.textContent = String(combinedActivity.length);
  if (countArtifactsEl) countArtifactsEl.textContent = String(workspaceArtifacts.length);

  if (tasksListEl) {
    if (combinedActivity.length === 0) {
      const hint = hermesStatus.configured
        ? 'Delegate coding, research, or app-building tasks by voice or via the bar above.'
        : 'Set HERMES_URL=http://127.0.0.1:8642 or HERMES_ENABLED=true to enable background delegation.';
      tasksListEl.innerHTML = `
        <div style="padding:24px 12px;text-align:center;color:var(--text-muted);font-size:12.5px;line-height:1.5;">
          <div style="margin-bottom:8px;color:var(--text-secondary);font-weight:500;">No background tasks</div>
          <div>${escapeHtml(hint)}</div>
        </div>
      `;
    } else {
      tasksListEl.innerHTML = combinedActivity.map((entry) => {
        if (entry.type === 'comfy') {
          const job = entry.item;
          const status = job.status || 'queued';
          const isRunning = status === 'running' || status === 'queued';
          const prog = job.progress;
          const progText = prog && prog.max ? `${prog.value || 0}/${prog.max} (${prog.percent || 0}%)` : '';
          return `
            <div class="hermes-task-card ${isRunning ? 'running' : ''}" data-job-id="${escapeHtml(job.job_id)}">
              <div class="hermes-task-header">
                <div style="min-width:0;flex:1;">
                  <div class="hermes-task-title">${escapeHtml(job.label || job.kind || 'ComfyUI')}</div>
                  <div class="hermes-task-meta">
                    <span class="task-status-tag ${escapeHtml(status)}">${escapeHtml(status)}</span>
                    <span>comfyui · ${escapeHtml(job.kind || 'media')}</span>
                    ${progText ? `<span>${escapeHtml(progText)}</span>` : ''}
                  </div>
                </div>
                <span style="color:var(--accent);display:inline-flex;">${window.SparkIcons.iconSvg(job.kind === 'music' ? 'music' : 'image', 14)}</span>
              </div>
              ${job.error ? `<div style="padding:8px 12px;font-size:12px;color:var(--danger);">${escapeHtml(job.error)}</div>` : ''}
              ${job.artifact ? `
                <div class="hermes-task-artifacts">
                  <button type="button" class="artifact-chip" onclick="previewWorkspaceArtifact(${JSON.stringify(job.artifact)})">
                    ${window.SparkIcons.iconSvg('file-image', 12)}
                    <span>${escapeHtml(job.artifact)}</span>
                  </button>
                </div>
              ` : ''}
            </div>
          `;
        }
        const task = entry.item;
        const status = task.status || 'queued';
        const isRunning = status === 'running' || status === 'queued';
        const steps = (task.steps || []).slice(-6);
        const artifacts = task.artifacts || [];
        const streamPreview = (task.output || task.thinking || '').trim();
        const dur = task.duration_ms ? `${(task.duration_ms / 1000).toFixed(1)}s` : '';

        return `
          <div class="hermes-task-card ${isRunning ? 'running' : ''}" data-task-id="${escapeHtml(task.task_id)}">
            <div class="hermes-task-header">
              <div style="min-width:0;flex:1;">
                <div class="hermes-task-title">${escapeHtml(task.task)}</div>
                <div class="hermes-task-meta">
                  <span class="task-status-tag ${escapeHtml(status)}">${escapeHtml(status)}</span>
                  <span>${escapeHtml(task.category || 'general')}</span>
                  ${dur ? `<span>${escapeHtml(dur)}</span>` : ''}
                </div>
              </div>
              ${isRunning ? `
                <button type="button" class="icon-btn" onclick="cancelHermesTask('${escapeHtml(task.task_id)}')" title="Cancel task" aria-label="Cancel task">
                  ${window.SparkIcons.iconSvg('square', 13)}
                </button>
              ` : ''}
            </div>

            ${steps.length > 0 ? `
              <div class="hermes-steps-list">
                ${steps.map((s) => `
                  <div class="hermes-step-item">
                    <span class="hermes-step-icon">${window.SparkIcons.iconSvg(s.type?.includes('failed') ? 'alert-circle' : 'terminal', 12)}</span>
                    <div style="min-width:0;flex:1;">
                      <div style="font-weight:500;color:var(--text-primary);">${escapeHtml(s.title || s.type)}</div>
                      ${s.detail ? `<div class="hermes-step-detail">${escapeHtml(s.detail)}</div>` : ''}
                    </div>
                  </div>
                `).join('')}
              </div>
            ` : ''}

            ${streamPreview ? `
              <div class="hermes-stream-box">${escapeHtml(streamPreview.slice(-900))}</div>
            ` : ''}

            ${task.error ? `
              <div style="padding:8px 12px;font-size:12px;color:var(--danger);background:var(--danger-muted);">
                ${escapeHtml(task.error)}
              </div>
            ` : ''}

            ${artifacts.length > 0 ? `
              <div class="hermes-task-artifacts">
                ${artifacts.map((relPath) => `
                  <button type="button" class="artifact-chip" onclick="previewWorkspaceArtifact(${JSON.stringify(relPath)})" title="Preview">
                    ${window.SparkIcons.iconSvg('file-code', 12)}
                    <span>${escapeHtml(relPath)}</span>
                  </button>
                  <button type="button" class="artifact-chip" onclick="openWorkspacePath(${JSON.stringify(relPath)}, false); event.stopPropagation();" title="Open on desktop">
                    ${window.SparkIcons.iconSvg('external-link', 12)}
                    <span>Open</span>
                  </button>
                `).join('')}
              </div>
            ` : ''}
          </div>
        `;
      }).join('');
    }
  }

  if (artifactsListEl) {
    if (workspaceArtifacts.length === 0) {
      artifactsListEl.innerHTML = `
        <div style="padding:24px 12px;text-align:center;color:var(--text-muted);font-size:12.5px;">
          Generated files in the shared workspace will appear here for live inspection and preview.
        </div>
      `;
    } else {
      artifactsListEl.innerHTML = workspaceArtifacts.map((art) => {
        const kb = art.size < 1024 ? `${art.size} B` : `${(art.size / 1024).toFixed(1)} KB`;
        const lower = art.name.toLowerCase();
        const isHtml = lower.endsWith('.html') || lower.endsWith('.htm');
        const isImage = /\.(png|jpe?g|webp|gif|svg)$/i.test(art.name);
        const isAudio = /\.(mp3|wav|flac|ogg)$/i.test(art.name);
        const iconName = isImage ? 'image' : (isAudio ? 'music' : (isHtml ? 'globe' : (lower.endsWith('.md') ? 'file-text' : 'file-code')));
        const pathEsc = escapeHtml(art.path);
        const pathJs = JSON.stringify(art.path);
        const mediaUrl = `/workspace/${art.path.split('/').map(encodeURIComponent).join('/')}`;
        const mediaPreview = isImage
          ? `<button type="button" class="artifact-media-thumb" onclick="openMediaLightbox(${pathJs})" title="Open full size">
              <img src="${mediaUrl}" alt="" loading="lazy" />
            </button>`
          : (isAudio ? `<div class="artifact-audio-wrap">${buildArtifactAudioPlayer(mediaUrl)}</div>` : '');
        return `
          <div class="artifact-row ${isImage || isAudio ? 'artifact-row-media' : ''}">
            ${mediaPreview}
            <div class="artifact-row-main" onclick="previewWorkspaceArtifact(${pathJs})" style="cursor:pointer;min-width:0;flex:1;">
              <span style="color:var(--accent);display:inline-flex;">${window.SparkIcons.iconSvg(iconName, 14)}</span>
              <span class="artifact-row-name">${pathEsc}</span>
            </div>
            <span class="artifact-row-meta">${escapeHtml(kb)}</span>
            <button type="button" class="icon-btn" onclick="event.stopPropagation(); openWorkspacePath(${pathJs}, false)" title="Open on desktop" aria-label="Open on desktop">
              ${window.SparkIcons.iconSvg('external-link', 13)}
            </button>
            <button type="button" class="icon-btn" onclick="event.stopPropagation(); openWorkspacePath(${pathJs}, true)" title="Reveal in folder" aria-label="Reveal in folder">
              ${window.SparkIcons.iconSvg('folder-open', 13)}
            </button>
          </div>
        `;
      }).join('');
    }
  }
}

async function openWorkspacePath(path, reveal = false) {
  if (!path) return;
  try {
    const res = await fetch('/api/workspace/open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, reveal: Boolean(reveal) })
    });
    const data = await res.json();
    if (!data.success) {
      addMessage('system', data.error || 'Failed to open path on host');
      return;
    }
    log(`Opened on host: ${data.opened || data.path}`);
  } catch (e) {
    log(`Open path error: ${e.message}`);
  }
}

async function previewPersonalFile(absPath) {
  try {
    const res = await fetch(`/api/personal/file?path=${encodeURIComponent(absPath)}`);
    const data = await res.json();
    if (!data.success || !data.file) {
      addMessage('system', data.error || 'Failed to preview personal file');
      return;
    }
    const file = data.file;
    if (file.binary) {
      await openWorkspacePath(file.path, false);
      return;
    }
    if (file.extension === '.html' || file.extension === '.htm') {
      openHtmlEditor(file.path);
      htmlContent = file.content || '';
      const editor = document.getElementById('agentHtmlEditor');
      const preview = document.getElementById('agentHtmlPreview');
      if (editor) editor.textContent = htmlContent;
      if (preview) preview.srcdoc = htmlContent;
    } else {
      openMarkdownEditor(file.path);
      markdownContent = file.content || '';
      const editor = document.getElementById('agentMarkdownEditor');
      const preview = document.getElementById('agentMarkdownPreview');
      if (editor) editor.textContent = markdownContent;
      if (preview) {
        if (file.extension === '.md') {
          preview.innerHTML = renderMarkdownPreview(markdownContent, 'agentMarkdownPreview');
        } else {
          preview.innerHTML = `<pre><code>${escapeHtml(markdownContent)}</code></pre>`;
        }
      }
    }
  } catch (e) {
    log(`Personal file preview error: ${e.message}`);
  }
}

async function submitHermesTaskFromUI() {
  const input = document.getElementById('hermesDelegateInput');
  const categorySelect = document.getElementById('hermesCategorySelect');
  const taskText = input?.value?.trim();
  if (!taskText) return;
  const category = categorySelect?.value || 'coding';
  input.value = '';

  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({
      type: 'delegate_task',
      task: taskText,
      category,
      speak_ack: false
    }));
    return;
  }

  try {
    const res = await fetch('/api/agents/delegate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: taskText, category })
    });
    const data = await res.json();
    if (data.success && data.task) {
      hermesTasks[data.task.task_id] = data.task;
      renderHermesPanel();
    } else if (data.error) {
      addMessage('system', data.error);
    }
  } catch (e) {
    log(`Delegate error: ${e.message}`);
  }
}

async function cancelHermesTask(taskId) {
  if (voiceWs && voiceWs.readyState === WebSocket.OPEN) {
    voiceWs.send(JSON.stringify({ type: 'cancel_hermes_task', task_id: taskId }));
    return;
  }
  try {
    await fetch(`/api/agents/tasks/${encodeURIComponent(taskId)}/cancel`, { method: 'POST' });
    await fetchHermesStatus();
  } catch (e) {
    log(`Cancel task error: ${e.message}`);
  }
}

function buildArtifactAudioPlayer(srcUrl) {
  const id = `audio-${Math.random().toString(36).slice(2, 9)}`;
  return `
    <div class="spark-audio-player">
      <audio id="${id}" preload="metadata" src="${escapeHtml(srcUrl)}"></audio>
      <button type="button" class="spark-audio-play" onclick="toggleSparkAudio('${id}', this)" aria-label="Play">
        ${window.SparkIcons.iconSvg('play', 14)}
      </button>
      <div class="spark-audio-track"><span class="spark-audio-fill"></span></div>
      <span class="spark-audio-time">0:00</span>
    </div>
  `;
}

function toggleSparkAudio(audioId, btn) {
  const audio = document.getElementById(audioId);
  if (!audio) return;
  if (audio.paused) {
    audio.play();
    btn.innerHTML = window.SparkIcons.iconSvg('pause', 14);
  } else {
    audio.pause();
    btn.innerHTML = window.SparkIcons.iconSvg('play', 14);
  }
}

function openMediaLightbox(relPath) {
  if (!relPath) return;
  const url = `/workspace/${relPath.split('/').map(encodeURIComponent).join('/')}`;
  const overlay = document.createElement('div');
  overlay.className = 'media-lightbox';
  overlay.innerHTML = `
    <div class="media-lightbox-inner">
      <button type="button" class="icon-btn media-lightbox-close" aria-label="Close">${window.SparkIcons.iconSvg('x', 16)}</button>
      <img src="${url}" alt="" class="media-lightbox-img" />
    </div>
  `;
  overlay.querySelector('.media-lightbox-close')?.addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
}

async function previewWorkspaceArtifact(relPath) {
  try {
    const res = await fetch(`/api/workspace/file?path=${encodeURIComponent(relPath)}`);
    const data = await res.json();
    if (!data.success || !data.file) return;
    const file = data.file;
    if (file.binary && file.workspace_url) {
      const ext = (file.extension || '').toLowerCase();
      if (/\.(png|jpe?g|webp|gif)$/i.test(ext)) {
        openMediaLightbox(file.path);
        return;
      }
      if (/\.(mp3|wav|flac|ogg)$/i.test(ext)) {
        switchAgentTab('artifacts');
        return;
      }
    }
    if (file.extension === '.html' || file.extension === '.htm') {
      openHtmlEditor(file.path);
      htmlContent = file.content;
      const editor = document.getElementById('agentHtmlEditor');
      const preview = document.getElementById('agentHtmlPreview');
      const charCount = document.getElementById('htmlCharCount');
      if (editor) editor.textContent = htmlContent;
      if (preview) preview.srcdoc = htmlContent;
      if (charCount) charCount.textContent = `${htmlContent.length} chars`;
      const statusEl = document.getElementById('htmlStatus');
      if (statusEl) statusEl.textContent = `Workspace artifact: ${file.path}`;
    } else {
      openMarkdownEditor(file.path);
      markdownContent = file.content;
      const editor = document.getElementById('agentMarkdownEditor');
      const preview = document.getElementById('agentMarkdownPreview');
      if (editor) editor.textContent = markdownContent;
      if (preview) {
        if (file.extension === '.md') {
          preview.innerHTML = renderMarkdownPreview(markdownContent, 'agentMarkdownPreview');
        } else {
          preview.innerHTML = `<pre><code>${escapeHtml(markdownContent)}</code></pre>`;
        }
      }
      const statusEl = document.getElementById('markdownStatus');
      if (statusEl) statusEl.textContent = `Workspace artifact: ${file.path}`;
    }
  } catch (e) {
    log(`Artifact preview error: ${e.message}`);
  }
}

// ============================================================================
// Markdown & HTML Live Workbench Modals + Inline Reasoning
// ============================================================================
function openMarkdownEditor(task) {
  const modal = document.getElementById('agentMarkdownEditorModal');
  const taskDesc = document.getElementById('markdownTaskDescription');
  const editor = document.getElementById('agentMarkdownEditor');
  const preview = document.getElementById('agentMarkdownPreview');

  markdownContent = '';
  if (editor) editor.textContent = '// Waiting for output...';
  if (preview) preview.innerHTML = '<span style="color:var(--text-muted);">Preview will render here...</span>';
  if (taskDesc) taskDesc.textContent = task || 'Document Preview';
  if (modal) {
    modal.classList.add('active');
    modal.style.display = 'flex';
  }
  markdownModalOpen = true;
}

function closeMarkdownEditor() {
  const modal = document.getElementById('agentMarkdownEditorModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
  markdownModalOpen = false;
}

function copyMarkdownToClipboard() {
  navigator.clipboard.writeText(markdownContent || '').then(() => {
    const btn = document.getElementById('copyMarkdownBtn');
    if (btn) {
      const prev = btn.innerHTML;
      btn.textContent = 'Copied';
      setTimeout(() => { btn.innerHTML = prev; }, 1500);
    }
  });
}

function createMarkdownMessageElement(task, markdown) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message message-assistant';

  const header = document.createElement('div');
  header.className = 'message-header';
  header.innerHTML = `<span>Markdown Output</span><span>${escapeHtml(task)}</span>`;

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.innerHTML = renderMarkdownPreview(markdown.substring(0, 1200) + (markdown.length > 1200 ? '\n\n...' : ''));

  messageDiv.appendChild(header);
  messageDiv.appendChild(contentDiv);
  return { container: messageDiv, content: contentDiv };
}

function openHtmlEditor(task) {
  const modal = document.getElementById('agentHtmlEditorModal');
  const taskDesc = document.getElementById('htmlTaskDescription');
  const editor = document.getElementById('agentHtmlEditor');
  const preview = document.getElementById('agentHtmlPreview');
  const charCount = document.getElementById('htmlCharCount');

  htmlContent = '';
  if (editor) editor.textContent = '// Waiting for HTML stream...';
  if (preview) preview.srcdoc = '<html><body style="font-family:sans-serif;color:#71717a;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">Preview loading...</body></html>';
  if (taskDesc) taskDesc.textContent = task || 'HTML Live Preview';
  if (charCount) charCount.textContent = '0 chars';
  if (modal) {
    modal.classList.add('active');
    modal.style.display = 'flex';
  }
  htmlModalOpen = true;
}

function closeHtmlEditor() {
  const modal = document.getElementById('agentHtmlEditorModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
  htmlModalOpen = false;
}

function refreshHtmlPreview() {
  const preview = document.getElementById('agentHtmlPreview');
  if (preview && htmlContent) preview.srcdoc = htmlContent;
}

function copyHtmlCode() {
  if (!htmlContent) return;
  navigator.clipboard.writeText(htmlContent);
}

function downloadHtml() {
  if (!htmlContent) return;
  const blob = new Blob([htmlContent], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'artifact.html';
  a.click();
  URL.revokeObjectURL(url);
}

// Inline Nemotron Reasoning Card
function createReasoningMessage(problem) {
  const container = document.createElement('div');
  container.className = 'reasoning-card';
  container.innerHTML = `
    <div class="reasoning-card-header">
      <span>Reasoning Trace (Nemotron) — ${escapeHtml(problem || 'Analysis')}</span>
      <span id="reasoningSpinner" class="task-status-tag running">Thinking</span>
    </div>
    <div class="reasoning-card-body">
      <div class="reasoning-trace" id="inlineThinkingContent"></div>
      <div id="inlineConclusionSection" style="display:none;">
        <div style="font-size:11px;font-weight:600;color:var(--accent);text-transform:uppercase;margin-bottom:4px;">Conclusion</div>
        <div id="inlineConclusionContent" style="font-size:13.5px;color:var(--text-primary);line-height:1.55;"></div>
      </div>
    </div>
  `;
  getActiveConversationEl().appendChild(container);
  scrollToBottom();

  currentReasoningContainer = container;
  currentThinkingEl = container.querySelector('#inlineThinkingContent');
  currentConclusionEl = container.querySelector('#inlineConclusionContent');
  inlineThinkingContent = '';
  inlineConclusionContent = '';
}

function appendInlineThinking(content) {
  if (!currentThinkingEl) return;
  inlineThinkingContent += content;
  currentThinkingEl.textContent = inlineThinkingContent;
  currentThinkingEl.scrollTop = currentThinkingEl.scrollHeight;
  scrollToBottom();
}

function appendInlineConclusion(content) {
  if (!currentConclusionEl) return;
  const section = currentReasoningContainer?.querySelector('#inlineConclusionSection');
  if (section) section.style.display = 'block';
  inlineConclusionContent += content;
  currentConclusionEl.textContent = inlineConclusionContent;
  scrollToBottom();
}

function finalizeReasoningMessage(finalContent) {
  const spinner = currentReasoningContainer?.querySelector('#reasoningSpinner');
  if (spinner) {
    spinner.className = 'task-status-tag completed';
    spinner.textContent = 'Completed';
  }
  if (finalContent && currentConclusionEl) {
    const section = currentReasoningContainer?.querySelector('#inlineConclusionSection');
    if (section) section.style.display = 'block';
    currentConclusionEl.textContent = finalContent;
  }
  currentReasoningContainer = null;
  currentThinkingEl = null;
  currentConclusionEl = null;
}
