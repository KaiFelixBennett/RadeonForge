/* Render the dark-aesthetic branding cards (HTML -> PNG) for the README + social preview.
 *   node render.mjs            -> writes assets/hero.png, assets/social-preview.png, assets/before-after.png
 */
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..');
const ASSETS = join(REPO, 'assets');
mkdirSync(ASSETS, { recursive: true });
const SRC = pathToFileURL(join(HERE, 'branding.html')).href;

const cards = [
  ['#hero', 'hero.png', 1280, 440],
  ['#social', 'social-preview.png', 1280, 640],
  ['#ba', 'before-after.png', 1200, 520],
];

const browser = await chromium.launch({ headless: true, args: ['--force-color-profile=srgb'] });
const page = await browser.newPage({ deviceScaleFactor: 2 });
await page.goto(SRC, { waitUntil: 'networkidle' });
await page.waitForFunction(() => document.body.dataset.ready === '1');
await page.waitForTimeout(400);
for (const [sel, name, w, h] of cards) {
  const el = page.locator(sel);
  await el.screenshot({ path: join(ASSETS, name) });
  console.log('  ' + name + '  (' + w + 'x' + h + ')');
}
await browser.close();
console.log('done ->', ASSETS);
