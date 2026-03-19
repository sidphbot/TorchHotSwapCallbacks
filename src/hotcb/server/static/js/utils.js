/**
 * hotcb dashboard — shared utilities
 */

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  try {
    const r = await fetch(path, opts);
    if (!r.ok) {
      console.warn('API', r.status, path);
      try { return await r.json(); } catch(_) { return null; }
    }
    return await r.json();
  } catch (e) { console.error('API error:', path, e); return null; }
}

function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }

function fmtNum(v, prec) {
  prec = prec || 4;
  if (v === null || v === undefined) return '--';
  if (Math.abs(v) < 0.01 || Math.abs(v) > 1e4) return v.toExponential(2);
  return v.toFixed(prec);
}

// Make closeModal globally available for inline onclick handlers
window.closeModal = closeModal;

// ---------------------------------------------------------------------------
// Panel loader helpers
// ---------------------------------------------------------------------------

/**
 * Show a loader overlay inside a container.
 * @param {string|Element} container - CSS selector or DOM element (must have position:relative)
 * @param {string} [text] - optional loading message
 * @returns {Element} the loader element (pass to hideLoader)
 */
function showLoader(container, text) {
  var el = typeof container === 'string' ? $(container) : container;
  if (!el) return null;
  // Ensure container has positioning for absolute overlay
  if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
  // Remove existing loader if any
  var existing = el.querySelector('.panel-loader');
  if (existing) existing.remove();
  var loader = document.createElement('div');
  loader.className = 'panel-loader';
  loader.innerHTML = '<div class="panel-loader-spinner"></div>' +
    (text ? '<div class="panel-loader-text">' + text + '</div>' : '');
  el.appendChild(loader);
  return loader;
}

/**
 * Hide (fade out and remove) a loader overlay.
 * @param {Element} loader - the element returned by showLoader
 */
function hideLoader(loader) {
  if (!loader) return;
  loader.classList.add('hidden');
  setTimeout(function() { if (loader.parentNode) loader.remove(); }, 400);
}

/**
 * Show an inline loader (small spinner + text, no overlay).
 * @param {string|Element} container - CSS selector or DOM element
 * @param {string} [text] - loading message
 * @returns {Element} the inline loader element
 */
function showInlineLoader(container, text) {
  var el = typeof container === 'string' ? $(container) : container;
  if (!el) return null;
  var loader = document.createElement('div');
  loader.className = 'inline-loader';
  loader.innerHTML = '<div class="inline-loader-spinner"></div>' +
    (text ? '<span class="inline-loader-text">' + text + '</span>' : '');
  el.innerHTML = '';
  el.appendChild(loader);
  return loader;
}
