/* RadeonForge — capture clean, brand-neutral dashboard assets from a running demo.
 *
 *   node capture.mjs <baseUrl> <outDir>
 *   node capture.mjs http://127.0.0.1:9137 ../../assets/dashboard
 *
 * Produces (into outDir):
 *   dashboard-gl.png         full /gl page (the hero)
 *   dashboard-home.png       full / canvas view
 *   panel-*.png              cropped panels (eval, hud, radar, scorecard, loss, tokens, results, flow)
 *   dashboard.gif            ~6 s live capture of the loss + tokens/s charts (frame loop -> ffmpeg)
 *
 * WebGL renders in headless Chromium via SwiftShader. The GIF needs a system `ffmpeg`
 * on PATH (the Playwright-bundled ffmpeg is webm-only); if absent, the GIF is skipped.
 */
import { chromium } from 'playwright';
import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const BASE = process.argv[2] || 'http://127.0.0.1:9137';
const OUT = process.argv[3] || 'E:/Coding/RadeonForge/assets/dashboard';
mkdirSync(OUT, { recursive: true });
const GL_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

const haveFfmpeg = spawnSync('ffmpeg', ['-version'], { encoding: 'utf-8' }).status === 0;

async function stills(browser) {
  const ctx = await browser.newContext({ viewport: { width: 1320, height: 1000 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto(BASE + '/gl', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5200);
  await page.screenshot({ path: join(OUT, 'dashboard-gl.png'), fullPage: true });
  console.log('  dashboard-gl.png');

  const panels = {
    'panel-eval': '#evalsec .card', 'panel-hud': '#hudsec .hud',
    'panel-radar': '#radarsec .card', 'panel-scorecard': '#scoresec .card',
    'panel-loss': '#losssec .card', 'panel-tokens': '#tpssec .card',
    'panel-results': '#chartsec .card', 'panel-flow': '#flowsec .card',
  };
  for (const [name, sel] of Object.entries(panels)) {
    const el = page.locator(sel).first();
    try { if (await el.count()) { await el.screenshot({ path: join(OUT, name + '.png') }); console.log('  ' + name + '.png'); } }
    catch (e) { console.log('  (skip ' + name + ')'); }
  }
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3500);
  await page.screenshot({ path: join(OUT, 'dashboard-home.png'), fullPage: true });
  console.log('  dashboard-home.png');
  await ctx.close();
}

async function gif(browser) {
  if (!haveFfmpeg) { console.log('  (skip dashboard.gif — no system ffmpeg on PATH)'); return; }
  const TMP = join(OUT, '_frames');
  rmSync(TMP, { recursive: true, force: true }); mkdirSync(TMP, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1180, height: 760 }, deviceScaleFactor: 1.5 });
  await page.goto(BASE + '/gl', { waitUntil: 'networkidle' });
  await page.waitForTimeout(4500);
  const clip = await page.evaluate(() => {
    const first = document.querySelector('#losssec');
    window.scrollTo(0, Math.max(0, first.getBoundingClientRect().top + window.scrollY - 14));
    const r = document.querySelector('.wrap').getBoundingClientRect();
    return { x: Math.max(0, r.left + 2), w: Math.min(1180, r.width - 4) };
  });
  await page.waitForTimeout(500);
  const N = 56;
  for (let i = 0; i < N; i++) {
    await page.screenshot({ path: join(TMP, `f${String(i).padStart(3, '0')}.png`), clip: { x: clip.x, y: 0, width: clip.w, height: 720 } });
    await page.waitForTimeout(120);
  }
  await page.close();
  const vf = 'fps=12,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3';
  const r = spawnSync('ffmpeg', ['-y', '-framerate', '12', '-i', join(TMP, 'f%03d.png'), '-vf', vf, '-loop', '0', join(OUT, 'dashboard.gif')], { encoding: 'utf-8' });
  if (r.status === 0) console.log('  dashboard.gif');
  else console.log('  ffmpeg failed:', (r.stderr || '').split('\n').slice(-3).join(' | '));
  rmSync(TMP, { recursive: true, force: true });
}

console.log('capture ->', OUT, 'from', BASE, haveFfmpeg ? '(ffmpeg ok)' : '(no ffmpeg)');
const browser = await chromium.launch({ headless: true, args: GL_ARGS });
await stills(browser);
await gif(browser);
await browser.close();
console.log('done.');
