/**
 * Sticky contents rail with scroll-spy.
 *
 * Marks the section nearest the top of the viewport as current. Using the
 * highest visible section rather than the largest visible one keeps the
 * highlight in step with reading order, which matters here because the figure
 * panels are much taller than the prose blocks between them.
 */

import { $$ } from './dom.js';

export function initNav() {
  const links = $$('.toc__link');
  if (!links.length) return;

  const sections = links
    .map((link) => ({ link, section: document.querySelector(link.getAttribute('href')) }))
    .filter((entry) => entry.section);

  let current = null;

  const setCurrent = (entry) => {
    if (entry === current) return;
    current?.link.removeAttribute('aria-current');
    entry?.link.setAttribute('aria-current', 'true');
    current = entry;

    // Keep the active item in view when the rail is a horizontal strip.
    if (entry && window.matchMedia('(max-width: 1080px)').matches) {
      entry.link.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
    }
  };

  const update = () => {
    const marker = window.innerHeight * 0.28;
    let best = sections[0];
    for (const entry of sections) {
      const top = entry.section.getBoundingClientRect().top;
      if (top <= marker) best = entry;
    }
    // At the very bottom the last section may never cross the marker.
    if (window.scrollY + window.innerHeight >= document.body.scrollHeight - 4) {
      best = sections[sections.length - 1];
    }
    setCurrent(best);
  };

  let ticking = false;
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { update(); ticking = false; });
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  update();
}
