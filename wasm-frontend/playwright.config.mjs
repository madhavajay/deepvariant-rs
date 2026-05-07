import { defineConfig, devices } from '@playwright/test';

// End-to-end smoke spec runs against Vite dev server.
export default defineConfig({
    testDir: './tests',
    timeout: 5 * 60 * 1000,
    expect: { timeout: 60_000 },
    fullyParallel: false,
    workers: 1,
    reporter: [['list']],
    use: {
        baseURL: 'http://127.0.0.1:5175',
        headless: true,
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure',
        navigationTimeout: 60_000,
    },
    webServer: {
        command: 'npx vite --port 5175 --strictPort --host 127.0.0.1',
        url: 'http://127.0.0.1:5175/',
        reuseExistingServer: false,
        timeout: 30_000,
        stdout: 'ignore',
        stderr: 'pipe',
    },
    projects: [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    ],
});
