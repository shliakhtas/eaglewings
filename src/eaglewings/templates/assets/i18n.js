// Дві мови сайту: українська (оригінал сторінок) і англійська (переклад зі словника i18n/en.json).
// Мова береться з адреси: /en/<сторінка> чи /ua/<сторінка>, або ?lang=en (мапа в iframe). Переклад іде по тексту сторінки
// й по всьому, що сторінка домалює пізніше (MutationObserver), тож працює й для застосунку, й для сторінки дослідження.
// Числа в рядку підставляються: «Знайдено 12 рядків» і «Знайдено 61 рядок» — один запис словника «Знайдено {0} рядків.».
(() => {
  const seg = location.pathname.split('/').filter(Boolean);
  const qs = new URLSearchParams(location.search).get('lang');
  const LANG = window.I18N_LANG || (seg.includes('en') ? 'en' : seg.includes('ua') ? 'ua' : (qs === 'en' ? 'en' : 'ua'));
  window.I18N_LANG = LANG;
  document.documentElement.lang = LANG === 'en' ? 'en' : 'uk';
  const inFrame = window.top !== window;

  // посилання між сторінками зберігають мову: /en/landing.html, а не /landing.html
  const ME = document.currentScript ? document.currentScript.src : location.href;
  const PAGE = /^\/(index\.html|landing\.html|app\.html|science\/(index\.html)?)?$/;
  function fixLinks(scope) {
    scope.querySelectorAll('a[href]').forEach(a => {
      const h = a.getAttribute('href'); if (!h || a.dataset.i18nLink || /^(#|mailto:|javascript:)/.test(h)) return;
      let u; try { u = new URL(h, document.baseURI); } catch (e) { return; }
      if (u.origin !== location.origin || !PAGE.test(u.pathname)) return;
      a.setAttribute('href', '/' + LANG + (u.pathname === '/' ? '/index.html' : u.pathname) + u.search + u.hash); a.dataset.i18nLink = '1';
    });
  }

  // перемикач мови вгорі праворуч
  function switcher() {
    if (inFrame || document.getElementById('i18n-switch')) return;
    const rest = location.pathname.replace(/^.*?\/(en|ua)\//, '').replace(/^\//, '') || 'index.html';
    const href = l => '/' + l + '/' + rest + location.search + location.hash;
    const d = document.createElement('div'); d.id = 'i18n-switch';
    d.innerHTML = `<a href="${href('ua')}" class="${LANG === 'ua' ? 'on' : ''}" hreflang="uk">UA</a><a href="${href('en')}" class="${LANG === 'en' ? 'on' : ''}" hreflang="en">EN</a>`;
    const st = document.createElement('style');
    st.textContent = '#i18n-switch{position:fixed;right:14px;top:12px;z-index:9999;display:flex;gap:2px;padding:3px;border-radius:999px;background:rgba(10,14,12,.86);border:1px solid rgba(255,255,255,.14);backdrop-filter:blur(6px);font:600 12px/1 Inter,system-ui,sans-serif}'
      + '#i18n-switch.inl{position:static;align-self:center;margin-left:14px;backdrop-filter:none}body > header #i18n-switch.inl{margin-left:auto}'
      + '#i18n-switch a{color:#c9cfc4;text-decoration:none;padding:6px 10px;border-radius:999px}#i18n-switch a.on{background:#4fd1c5;color:#06201c}#i18n-switch a:not(.on):hover{color:#fff}';
    document.head.appendChild(st);
    const slot = document.querySelector('[data-i18n-slot]') || document.querySelector('nav .links') || document.querySelector('body > header');
    if (slot) { d.classList.add('inl'); slot.appendChild(d); } else document.body.appendChild(d);
  }

  if (LANG !== 'en') { addEventListener('DOMContentLoaded', () => { fixLinks(document); switcher(); }); return; }

  // ---------- переклад
  const NUM = /\d+(?:[ \u00a0\u202f][\d]{3})*(?:[.,]\d+)?/g;
  let DICT = {}, PAT = {};
  const norm = s => { let i = 0; const nums = []; const k = s.replace(NUM, m => { nums.push(m); return '{' + (i++) + '}'; }); return [k, nums]; };
  const enNum = n => n.replace(/[ \u00a0\u202f](?=\d{3})/g, ',').replace(/(\d),(\d{1,2})$/, '$1.$2').replace(/^(\d+),(\d+)$/, (m, a, b) => b.length === 3 ? m : a + '.' + b);
  function tr(s) {
    const t = s.trim(); if (!t || !/[А-Яа-яЄєІіЇїҐґ]/.test(t)) return null;
    let out = DICT[t];
    if (out === undefined) { const [k, nums] = norm(t); const p = PAT[k]; if (p === undefined) return null;
      out = p.replace(/\{(\d+)\}/g, (m, i) => nums[+i] !== undefined ? enNum(nums[+i]) : m); }
    return s.replace(t, out);
  }
  const ATTR = ['placeholder', 'title', 'alt', 'aria-label'];
  function walk(n) {
    if (n.nodeType === 3) { const r = tr(n.nodeValue); if (r !== null && r !== n.nodeValue) n.nodeValue = r; return; }
    if (n.nodeType !== 1 || n.tagName === 'SCRIPT' || n.tagName === 'STYLE') return;
    for (const a of ATTR) { const v = n.getAttribute(a); if (v) { const r = tr(v); if (r !== null && r !== v) n.setAttribute(a, r); } }
    if (n.tagName === 'INPUT' && n.type === 'button' && n.value) { const r = tr(n.value); if (r) n.value = r; }
    n.childNodes.forEach(walk);
  }
  const hide = document.createElement('style'); hide.id = 'i18n-hide'; hide.textContent = 'body{visibility:hidden}'; document.head.appendChild(hide);
  const show = () => { const h = document.getElementById('i18n-hide'); if (h) h.remove(); };
  setTimeout(show, 2500);                                      // словник не завантажився — показуємо як є
  const dictUrl = new URL('../i18n/en.json', ME).href;
  const ready = fetch(dictUrl).then(r => r.json()).then(j => { DICT = j.exact || {}; PAT = j.patterns || {}; }).catch(() => {});
  addEventListener('DOMContentLoaded', () => ready.then(() => {
    if (document.title) { const r = tr(document.title); if (r) document.title = r; }
    walk(document.body); fixLinks(document); switcher(); show();
    new MutationObserver(ms => ms.forEach(m => {
      if (m.type === 'characterData') { const r = tr(m.target.nodeValue); if (r !== null && r !== m.target.nodeValue) m.target.nodeValue = r; }
      else if (m.type === 'attributes') walk(m.target);
      else m.addedNodes.forEach(x => { walk(x); if (x.nodeType === 1) fixLinks(x); });
    })).observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTR});
  }));
  window.I18N_T = s => { const r = tr(String(s)); return r === null ? s : r; };   // для тексту, що йде не в DOM (графіки, підказки мап)
})();
