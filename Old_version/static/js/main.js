/**
 * main.js — PulseMailer Enhanced Client
 * ====================================
 * SOCKET ADDITIONS (Part 1):
 *   - initSocket()         : Connects to Flask-SocketIO server
 *   - Typing indicator     : Emits 'typing' event while user types in compose
 *   - Online/offline status: Listens to 'online_users' and marks users
 *   - Live inbox update    : Listens to 'new_message' and prepends row
 *   - Notifications        : Shows bell badge + dropdown on new message
 *
 * All original functions (flash, sidebar, select-all, autocomplete,
 * thread toggle, password toggle, colour picker, char count) unchanged.
 */

/* ── DOM Ready ─────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initFlashMessages();
  initMobileSidebar();
  initSelectAll();
  initAutocomplete();
  initThreadToggle();
  initPasswordToggle();
  initColorPicker();
  initComposeCharCount();
  refreshBadges();
  initSocket();       // NEW: Real-time socket connection
  initNotifBell();    // NEW: Notification bell toggle
});

/* ════════════════════════════════════════════════════════════════════
   SOCKET.IO INTEGRATION (Part 1)
   ════════════════════════════════════════════════════════════════════ */

let socket = null;
let _onlineUsers = new Set();
let _notifications = [];    // in-memory list of recent notifications
let _typingTimer = null;

function initSocket() {
  // Only connect if the user is logged in (sidebar exists)
  if (!document.querySelector('.sidebar')) return;

  // Load Socket.IO client library injected by Flask-SocketIO
  // The library is served automatically at /socket.io/socket.io.js
  if (typeof io === 'undefined') {
    console.warn('[Socket] io not available — SocketIO client not loaded');
    return;
  }

  // ── Connect ────────────────────────────────────────────────────────
  socket = io();   // connects to same origin
    window._socket = socket;

  socket.on('connect', () => {
    console.log('[Socket] Connected:', socket.id);
  });

  socket.on('disconnect', () => {
    console.log('[Socket] Disconnected');
    _onlineUsers.clear();
    refreshOnlineUI();
  });

  // ── Online / Offline status (Part 1.4) ────────────────────────────
  socket.on('online_users', (data) => {
    _onlineUsers = new Set(data.users || []);
    refreshOnlineUI();
  });

  // ── Typing indicator (Part 1.3) ───────────────────────────────────
  socket.on('typing', (data) => {
    const indicator = document.getElementById('typingIndicator');
    if (!indicator) return;
    if (data.is_typing) {
      indicator.querySelector('.typing-name').textContent = data.username;
      indicator.style.display = 'flex';
    } else {
      indicator.style.display = 'none';
    }
  });

  // ── Live inbox update + notification (Part 1.2 & 1.5) ─────────────
  socket.on('new_message', (data) => {
    // Play a subtle audio cue (browser permitting)
    playNotifSound();

    // Add to in-memory notification list
    _notifications.unshift(data);
    if (_notifications.length > 10) _notifications.pop();
    renderNotifDropdown();
    showNotifBadge(_notifications.length);

    const path = window.location.pathname;
    if (data.message_id) {
      const list = document.querySelector('.message-list');
      if (list && list.querySelector(`.message-row[data-message-id="${data.message_id}"]`)) {
        return; // avoid duplicate render when same event arrives twice
      }
    }

    if (path === '/inbox' && data.folder === 'inbox') {
      prependMessageRow(data);
    }
    if (path === '/spam' && data.folder === 'spam') {
      prependMessageRow(data);
    }

    // Update sidebar badges based on actual folder
    const inboxBadge = document.getElementById('inboxBadge');
    const spamBadge  = document.getElementById('spamBadge');
    if (data.folder === 'inbox' && inboxBadge) {
      const n = parseInt(inboxBadge.textContent || '0') + 1;
      inboxBadge.textContent = n;
      inboxBadge.style.display = '';
    }
    if (data.folder === 'spam' && spamBadge) {
      const n = parseInt(spamBadge.textContent || '0') + 1;
      spamBadge.textContent = n;
      spamBadge.style.display = '';
    }
  });
}

/* ── Emit typing event from compose textarea ─────────────────────── */
function initTypingEmit(recipientId) {
  if (!socket || !recipientId) return;
  const body = document.getElementById('messageBody');
  if (!body) return;

  const username = document.querySelector('.app-shell')?.dataset.currentUsername || 'Someone';

  body.addEventListener('input', () => {
    socket.emit('typing', { recipient_id: parseInt(recipientId), username, is_typing: true });
    clearTimeout(_typingTimer);
    _typingTimer = setTimeout(() => {
      socket.emit('typing', { recipient_id: parseInt(recipientId), username, is_typing: false });
    }, 2000);
  });
}

/* ── Prepend a new message row to the inbox list (Part 1.2) ─────── */
function prependMessageRow(data) {
  const list = document.querySelector('.message-list');
  if (!list) return;

  const row = document.createElement('div');
  row.className = 'message-row unread new-arrive';
  if (data.message_id) {
    row.dataset.messageId = data.message_id;
    row.addEventListener('click', () => location.href = `/message/${data.message_id}`);
  }
  row.style.cursor = 'pointer';

  const color = data.sender_color || '#528fff';
  const initial = (data.sender_display || 'S')[0].toUpperCase();
  const now = new Date().toISOString().slice(0,10);

  row.innerHTML = `
    <span class="unread-dot"></span>
    <div class="msg-checkbox"><input type="checkbox" class="msg-check" disabled /></div>
    <div class="msg-from">
      <div class="d-flex align-center gap-8">
        <div class="avatar-sm" style="background:${escHtml(color)};width:24px;height:24px;font-size:.7rem">${escHtml(initial)}</div>
        ${escHtml(data.sender_display || 'Someone')}
      </div>
    </div>
    <div class="msg-preview">
      <span class="msg-subject">${escHtml(data.subject || '(no subject)')}</span>
      <span class="msg-snippet">${escHtml(data.preview || '')}</span>
    </div>
    <div class="msg-icons"></div>
    <div class="msg-time">${escHtml(now)}</div>
  `;

  list.prepend(row);
}

/* ── Refresh online indicator dots in compose autocomplete ─────── */
function refreshOnlineUI() {
  document.querySelectorAll('[data-user-id]').forEach(el => {
    const uid = parseInt(el.dataset.userId);
    const dot = el.querySelector('.online-dot');
    if (dot) dot.style.display = _onlineUsers.has(uid) ? '' : 'none';
  });
}

/* ── Notification bell ─────────────────────────────────────────── */
function initNotifBell() {
  const bell     = document.getElementById('notifBell');
  const dropdown = document.getElementById('notifDropdown');
  if (!bell || !dropdown) return;

  bell.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('open');
    if (dropdown.classList.contains('open')) {
      showNotifBadge(0);   // clear badge on open
    }
  });
  document.addEventListener('click', () => dropdown.classList.remove('open'));
}

function showNotifBadge(n) {
  const el = document.getElementById('notifCount');
  if (!el) return;
  if (n > 0) {
    el.textContent = n > 9 ? '9+' : n;
    el.style.display = 'flex';
  } else {
    el.style.display = 'none';
  }
}

function renderNotifDropdown() {
  const dropdown = document.getElementById('notifDropdown');
  if (!dropdown) return;

  if (_notifications.length === 0) {
    dropdown.innerHTML = '<div class="notif-header">Notifications</div><div class="notif-empty">No new messages</div>';
    return;
  }

  let html = '<div class="notif-header">New Messages</div>';
  _notifications.forEach(n => {
    const color = n.sender_color || '#528fff';
    const initial = (n.sender_display || 'S')[0].toUpperCase();
    const href = n.message_id ? `/message/${n.message_id}` : '/inbox';
    html += `
      <div class="notif-item" onclick="location.href='${href}'">
        <div class="notif-avatar" style="background:${escHtml(color)}">${escHtml(initial)}</div>
        <div class="notif-body">
          <div class="notif-sender">${escHtml(n.sender_display || 'Someone')}</div>
          <div class="notif-subject">${escHtml(n.subject || '(no subject)')}</div>
          <div class="notif-time">Just now</div>
        </div>
      </div>`;
  });

  dropdown.innerHTML = html;
}

/* ── Subtle notification sound (Web Audio API) ───────────────────── */
function playNotifSound() {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.3);
  } catch (_) { /* ignore if blocked */ }
}

/* ════════════════════════════════════════════════════════════════════
   ORIGINAL FUNCTIONS (unchanged)
   ════════════════════════════════════════════════════════════════════ */

/* ── Flash message auto-dismiss ──────────────────────────────────── */
function initFlashMessages() {
  document.querySelectorAll('.flash .close-btn').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('.flash').remove());
  });
  document.querySelectorAll('.flash.success, .flash.info').forEach(el => {
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity .3s';
      setTimeout(() => el.remove(), 300);
    }, 4000);
  });
}

/* ── Mobile sidebar toggle ───────────────────────────────────────── */
function initMobileSidebar() {
  const btn     = document.getElementById('mobileMenuBtn');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  if (!btn || !sidebar) return;
  const open  = () => { sidebar.classList.add('open');  overlay?.classList.add('visible'); };
  const close = () => { sidebar.classList.remove('open'); overlay?.classList.remove('visible'); };
  btn.addEventListener('click', open);
  overlay?.addEventListener('click', close);
}

/* ── Select-all checkbox ─────────────────────────────────────────── */
function initSelectAll() {
  const master = document.getElementById('selectAll');
  if (!master) return;
  const children = () => document.querySelectorAll('.msg-check');
  master.addEventListener('change', () => {
    children().forEach(cb => cb.checked = master.checked);
    toggleBulkBar();
  });
  document.addEventListener('change', e => {
    if (e.target.classList.contains('msg-check')) toggleBulkBar();
  });
  function toggleBulkBar() {
    const bar = document.getElementById('bulkBar');
    if (!bar) return;
    const any = [...children()].some(cb => cb.checked);
    bar.style.display = any ? 'flex' : 'none';
  }
}

/* ── Autocomplete for "To" field ─────────────────────────────────── */
function initAutocomplete() {
  const input = document.getElementById('toInput');
  const list  = document.getElementById('autocompleteList');
  if (!input || !list) return;

  let hIdx = -1, debounce;

  input.addEventListener('input', () => {
    clearTimeout(debounce);
    const q = input.value.trim();
    if (q.length < 1) { list.innerHTML = ''; list.style.display = 'none'; return; }
    debounce = setTimeout(() => fetchUsers(q), 200);
  });

  input.addEventListener('keydown', e => {
    const items = list.querySelectorAll('.autocomplete-item');
    if (e.key === 'ArrowDown') { e.preventDefault(); hIdx = Math.min(hIdx + 1, items.length - 1); highlight(items); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); hIdx = Math.max(hIdx - 1, -1); highlight(items); }
    else if (e.key === 'Enter' && hIdx >= 0) { e.preventDefault(); items[hIdx]?.click(); }
    else if (e.key === 'Escape') { list.style.display = 'none'; }
  });

  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !list.contains(e.target)) list.style.display = 'none';
  });

  function highlight(items) {
    items.forEach((it, i) => it.classList.toggle('highlighted', i === hIdx));
  }

  async function fetchUsers(q) {
    try {
      const res  = await fetch(`/api/users/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      list.innerHTML = '';
      hIdx = -1;
      if (!data.length) { list.style.display = 'none'; return; }
      data.forEach(u => {
        const item = document.createElement('div');
        item.className = 'autocomplete-item';
        item.dataset.userId = u.id;
        const isOnline = _onlineUsers.has(u.id);
        item.innerHTML = `<span class="user-name">${escHtml(u.username)}</span>
                          <span class="display-name"> — ${escHtml(u.display_name || u.username)}</span>
                          ${isOnline ? '<span class="user-online-badge" title="Online"></span>' : ''}`;
        item.addEventListener('click', () => {
          input.value = u.username;
          list.style.display = 'none';
          if (u.id) initTypingEmit(u.id);
        });
        list.appendChild(item);
      });
      list.style.display = 'block';
    } catch (_) { /* ignore */ }
  }
}

/* ── Thread message expand/collapse ─────────────────────────────── */
function initThreadToggle() {
  document.querySelectorAll('.thread-msg-header').forEach(header => {
    header.addEventListener('click', () => {
      const body = header.nextElementSibling;
      body?.classList.toggle('open');
      const chevron = header.querySelector('.thread-chevron');
      if (chevron) chevron.style.transform = body?.classList.contains('open') ? 'rotate(180deg)' : '';
    });
  });
}

/* ── Password visibility toggle ─────────────────────────────────── */
function initPasswordToggle() {
  document.querySelectorAll('.password-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const inp = btn.previousElementSibling;
      if (!inp) return;
      const show = inp.type === 'password';
      inp.type = show ? 'text' : 'password';
      btn.textContent = show ? '🙈' : '👁';
    });
  });
}

/* ── Avatar colour picker ────────────────────────────────────────── */
function initColorPicker() {
  const hidden  = document.getElementById('avatarColorInput');
  const swatches = document.querySelectorAll('.color-swatch');
  if (!hidden || !swatches.length) return;
  swatches.forEach(sw => {
    if (sw.dataset.color === hidden.value) sw.classList.add('selected');
    sw.addEventListener('click', () => {
      swatches.forEach(s => s.classList.remove('selected'));
      sw.classList.add('selected');
      hidden.value = sw.dataset.color;
      const preview = document.getElementById('avatarPreview');
      if (preview) preview.style.background = sw.dataset.color;
    });
  });
}

/* ── Compose body char count ─────────────────────────────────────── */
function initComposeCharCount() {
  const body  = document.getElementById('messageBody');
  const count = document.getElementById('charCount');
  if (!body || !count) return;
  const update = () => { count.textContent = `${body.value.length} chars`; };
  body.addEventListener('input', update);
  update();
}

/* ── Poll unread badge counts every 30s ─────────────────────────── */
function refreshBadges() {
  if (!document.querySelector('.sidebar')) return;
  setInterval(async () => {
    try {
      const res  = await fetch('/api/folder-counts');
      const data = await res.json();
      updateBadge('inboxBadge', data.inbox);
      updateBadge('spamBadge',  data.spam);
    } catch (_) { /* ignore */ }
  }, 30_000);
}

function updateBadge(id, n) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = n > 0 ? n : '';
  el.style.display = n > 0 ? '' : 'none';
}

/* ── Utility ─────────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
