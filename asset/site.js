document.documentElement.classList.add('has-js');

const avatarImage = document.querySelector('#avatar-image');
if (avatarImage) {
  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  let livePortrait;
  let loadingPortrait;
  async function loadLivePortrait() {
    if (livePortrait || motionPreference.matches) return;
    if (loadingPortrait) return loadingPortrait;
    loadingPortrait = (async () => {
      try {
        const response = await fetch(avatarImage.dataset.animated);
        if (!response.ok) throw new Error('Avatar unavailable');
        const document = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
        if (document.querySelector('parsererror') || document.documentElement.localName !== 'svg') {
          throw new Error('Invalid avatar');
        }
        const portrait = window.document.importNode(document.documentElement, true);
        portrait.classList.add('avatar-live');
        portrait.setAttribute('aria-label', avatarImage.alt);
        portrait.setAttribute('focusable', 'false');
        avatarImage.closest('a').append(portrait);
        livePortrait = portrait;
        portrait.setCurrentTime(0);
        updateAvatarMotion();
      } catch {
        // Keep the static portrait when animation cannot be loaded.
      }
    })();
    return loadingPortrait;
  }
  function updateAvatarMotion() {
    if (livePortrait) {
      if (motionPreference.matches) {
        livePortrait.pauseAnimations();
        livePortrait.setAttribute('hidden', '');
        avatarImage.hidden = false;
      } else {
        livePortrait.removeAttribute('hidden');
        avatarImage.hidden = true;
        livePortrait.unpauseAnimations();
      }
    } else if (!motionPreference.matches) {
      loadLivePortrait();
    }
  }
  motionPreference.addEventListener('change', updateAvatarMotion);
  updateAvatarMotion();
}

const search = document.querySelector('#publication-search');
if (search) {
  const year = document.querySelector('#publication-year');
  const type = document.querySelector('#publication-type');
  const papers = [...document.querySelectorAll('.pub[data-search]')];
  const groups = [...document.querySelectorAll('[data-publication-group]')];
  const status = document.querySelector('#result-status');
  const empty = document.querySelector('#no-results');
  const yearDetails = [...document.querySelectorAll('.year-more')];
  const total = new Set(papers.map(paper => paper.dataset.paperId)).size;
  let filtering = false;
  const savedOpen = new Map();
  function filter() {
    const terms = search.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    const active = Boolean(terms.length || year.value || type.value);
    if (active && !filtering) yearDetails.forEach(details => savedOpen.set(details, details.open));
    const visible = new Set();
    papers.forEach(paper => {
      const matches = terms.every(term => paper.dataset.search.includes(term)) &&
        (!year.value || year.value === paper.dataset.year) &&
        (!type.value || type.value === paper.dataset.type || type.value === paper.dataset.status);
      paper.hidden = !matches;
      if (matches) visible.add(paper.dataset.paperId);
    });
    yearDetails.forEach(details => {
      details.hidden = !details.querySelector('.pub:not([hidden])');
      if (active) details.open = !details.hidden;
      else if (filtering) details.open = savedOpen.get(details) || false;
    });
    groups.forEach(group => { group.hidden = !group.querySelector('.pub:not([hidden])'); });
    status.textContent = `${visible.size} of ${total} publications`;
    empty.hidden = visible.size !== 0;
    filtering = active;
  }
  [search, year, type].forEach(input => input.addEventListener('input', filter));
  filter();
  function revealPaper() {
    let id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target?.matches('.pub')) return;
    if (target.hidden) {
      search.value = year.value = type.value = '';
      filter();
    }
    const details = target.closest('.year-more');
    if (details) details.open = true;
    requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  }
  window.addEventListener('hashchange', revealPaper);
  revealPaper();
}

document.querySelectorAll('.copy-btn').forEach(button => {
  button.addEventListener('click', async () => {
    const text = button.closest('.citation-box').querySelector('pre').textContent;
    const message = button.nextElementSibling;
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(text);
      message.textContent = 'BibTeX copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(button.closest('.citation-box').querySelector('pre'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      message.textContent = 'BibTeX selected — copy with Ctrl+C or ⌘C';
    }
  });
});
