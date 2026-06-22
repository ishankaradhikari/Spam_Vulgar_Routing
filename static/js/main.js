/**
 * main.js — PulseMailer client-side logic
 * Handles: flash dismiss, mobile sidebar, select-all, autocomplete,
 *          thread expand/collapse, password toggle, char count, colour picker
 */

/* ── DOM Ready ───────────────────────────────────────────────────── */
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
});

/* ── Flash message auto-dismiss ──────────────────────────────────── */
function initFlashMessages() {
  document.querySelectorAll('.flash .close-btn').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('.flash').remove());
  });

  // Auto-dismiss success/info flashes after 4s
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

  let hIdx = -1;
  let debounce;

  input.addEventListener('input', () => {
    clearTimeout(debounce);
    const q = input.value.trim();
    if (q.length < 1) { list.innerHTML = ''; list.style.display = 'none'; return; }
    debounce = setTimeout(() => fetchUsers(q), 200);
  });

  input.addEventListener('keydown', e => {
    const items = list.querySelectorAll('.autocomplete-item');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      hIdx = Math.min(hIdx + 1, items.length - 1);
      highlight(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      hIdx = Math.max(hIdx - 1, -1);
      highlight(items);
    } else if (e.key === 'Enter' && hIdx >= 0) {
      e.preventDefault();
      items[hIdx]?.click();
    } else if (e.key === 'Escape') {
      list.style.display = 'none';
    }
  });

  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !list.contains(e.target)) {
      list.style.display = 'none';
    }
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
        item.innerHTML = `<span class="user-name">${escHtml(u.username)}</span>
                          <span class="display-name"> — ${escHtml(u.display_name || u.username)}</span>`;
        item.addEventListener('click', () => {
          input.value      = u.username;
          list.style.display = 'none';
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
      // Live update profile avatar preview
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

/* ── Poll unread badge counts every 60s ──────────────────────────── */
function refreshBadges() {
  if (!document.querySelector('.sidebar')) return;

  setInterval(async () => {
    try {
      const res  = await fetch('/api/folder-counts');
      const data = await res.json();
      updateBadge('inboxBadge', data.inbox);
      updateBadge('spamBadge',  data.spam);
    } catch (_) { /* ignore */ }
  }, 60_000);
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
