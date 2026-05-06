import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    timeout: 5 * 60 * 1000,
    expect: { timeout: 60_000 },
    fullyParallel: false,
    workers: 1,
    reporter: [['list']],
    use: {
        baseURL: 'http://127.0.0.1:8089',
        headless: true,
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure',
        // The WGS .onnx is 87MB; needs the navigation timeout above.
        navigationTimeout: 60_000,
    },
    webServer: {
        command: 'python3 -m http.server 8089 --bind 127.0.0.1',
        url: 'http://127.0.0.1:8089/index.html',
        cwd: 'public',
        reuseExistingServer: false,
        timeout: 30_000,
        stdout: 'ignore',
        stderr: 'pipe',
    },
    projects: [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    ],
});
