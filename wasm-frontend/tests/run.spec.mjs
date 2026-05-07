// End-to-end smoke for wasm-frontend.
//
// Walks the same flow a user would:
//   1. Click "Load WASM"                       → wasm-status badge flips to "loaded"
//   2. Click "Download model"                  → model-status badge flips to "cached"
//   3. Click "Run on bundled chr20 test sample" → results panel appears
//   4. Click "Download .tar.gz"                → archive saves with the expected files
//
// We also assert no console errors throughout.

import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('full flow: load wasm, download model, run sample, download tar.gz', async ({ page }) => {
    const errs = [];
    page.on('pageerror', (e) => errs.push(`pageerror: ${e.message}`));
    page.on('console', (msg) => {
        if (msg.type() === 'error') errs.push(`console.error: ${msg.text()}`);
    });

    await page.goto('/');

    // ── Step 1: WASM ──
    await page.click('#btn-load-wasm');
    await expect(page.locator('#wasm-badge')).toHaveText(/loaded/, { timeout: 30_000 });

    // ── Step 2: model ──
    await page.click('#btn-download-model');
    await expect(page.locator('#model-badge')).toHaveText(/cached/, { timeout: 4 * 60 * 1000 });

    // ── Step 3: run ──
    await page.click('#btn-run-sample');
    await expect(page.locator('#run-status')).toContainText('done', { timeout: 4 * 60 * 1000 });

    // Results panel should now be populated.
    await expect(page.locator('#panel-results')).toBeVisible();
    const rowCount = await page.locator('#results tbody tr').count();
    expect(rowCount).toBe(32); // chr20 norealign fixture has 32 examples

    // Spot-check probability columns are non-empty.
    const firstRow = page.locator('#results tbody tr').first();
    await expect(firstRow.locator('td').nth(4)).not.toBeEmpty();
    await expect(firstRow.locator('td').nth(5)).not.toBeEmpty();
    await expect(firstRow.locator('td').nth(6)).not.toBeEmpty();

    // ── Step 4: download bundle ──
    const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 30_000 }),
        page.click('#btn-download'),
    ]);
    expect(download.suggestedFilename()).toMatch(/\.tar\.gz$/);

    const dir = mkdtempSync(join(tmpdir(), 'dv-tar-'));
    const path = join(dir, download.suggestedFilename());
    await download.saveAs(path);

    // Verify the archive is well-formed and contains the expected files.
    const list = execFileSync('tar', ['-tzf', path], { encoding: 'utf8' });
    expect(list).toMatch(/\/predictions\.json$/m);
    expect(list).toMatch(/\/summary\.txt$/m);
    expect(list).toMatch(/\/README\.md$/m);

    // Extract predictions.json and check it has 32 entries.
    execFileSync('tar', ['-xzf', path, '-C', dir]);
    const dirName = list.split('\n')[0].split('/')[0];
    const preds = JSON.parse(readFileSync(join(dir, dirName, 'predictions.json'), 'utf8'));
    expect(preds.predictions).toHaveLength(32);
    expect(preds.meta.example_count).toBe(32);
    // Each prediction should carry a chrom and probabilities sum near 1.
    for (const p of preds.predictions) {
        expect(p.chrom).toBeTruthy();
        const s = p.probabilities.hom_ref + p.probabilities.het + p.probabilities.hom_alt;
        expect(s).toBeGreaterThan(0.99);
        expect(s).toBeLessThan(1.01);
    }

    expect(errs, `unexpected page errors:\n${errs.join('\n')}`).toHaveLength(0);
});
