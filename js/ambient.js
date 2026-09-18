/**
 * A slow heartbeat for a read-only figure.
 *
 * Calls `act` every `every` milliseconds while the figure is on screen and
 * the tab is visible, waits for `act` to settle before counting the interval,
 * and stops for good the first time the reader touches the figure or any
 * control in its section — from then on the figure is theirs. Under reduced
 * motion it never starts.
 */
export function ambient({ root, canvas, every, act }) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let stopped = reduced, paused = false, onScreen = false, timer = 0, busy = false;

  const clear = () => { clearTimeout(timer); timer = 0; };
  const schedule = () => {
    clear();
    if (stopped || paused || !onScreen || document.hidden || busy) return;
    timer = setTimeout(async () => {
      timer = 0;
      if (stopped || paused || !onScreen || document.hidden) return;
      busy = true;
      try { await act(); } finally { busy = false; }
      schedule();
    }, every);
  };
  const stop = () => { stopped = true; clear(); };
  const pause = () => { paused = true; clear(); };
  const resume = () => { paused = false; schedule(); };

  if (!reduced) {
    new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; schedule(); }, { threshold: 0.3 }).observe(canvas);
    document.addEventListener('visibilitychange', schedule);
    for (const type of ['pointerdown', 'keydown']) canvas.addEventListener(type, stop);
    for (const el of root.querySelectorAll('select, input, button')) {
      el.addEventListener('change', stop);
      el.addEventListener('click', stop);
    }
  }
  return { stop, pause, resume };
}
