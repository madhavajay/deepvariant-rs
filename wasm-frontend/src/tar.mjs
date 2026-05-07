// Tiny USTAR (POSIX 1003.1-1990) tar archive builder for the browser.
//
// Each file is one 512-byte header + payload + zero-padding to the
// next 512-byte boundary. The archive ends with two zeroed 512-byte
// blocks. We don't need any of the GNU/PAX extensions for the small
// JSON/text/binary blobs we ship.

const BLOCK = 512;

function padOctal(n, width) {
    // POSIX wants right-justified octal with leading zeros, NUL-terminated.
    return n.toString(8).padStart(width - 1, '0') + '\0';
}

function writeStr(buf, off, s, max) {
    const enc = new TextEncoder().encode(s);
    if (enc.length > max) throw new Error(`tar field overflow: "${s}"`);
    buf.set(enc, off);
}

function buildHeader({ name, size, mtime = Math.floor(Date.now() / 1000), mode = 0o644 }) {
    const h = new Uint8Array(BLOCK);
    if (name.length > 100) throw new Error(`filename too long for ustar: ${name}`);
    writeStr(h, 0, name, 100);
    writeStr(h, 100, padOctal(mode, 8), 8);
    writeStr(h, 108, padOctal(0, 8), 8);              // uid
    writeStr(h, 116, padOctal(0, 8), 8);              // gid
    writeStr(h, 124, padOctal(size, 12), 12);
    writeStr(h, 136, padOctal(mtime, 12), 12);
    // Checksum field: spec says treat the 8-byte field as ASCII
    // spaces during the sum. Write spaces in, sum the whole header,
    // then overwrite with the actual checksum.
    h.set(new TextEncoder().encode('        '), 148);
    h[156] = 0x30;                                    // typeflag '0' (regular file)
    writeStr(h, 257, 'ustar\0', 6);
    writeStr(h, 263, '00', 2);                        // version

    let sum = 0;
    for (let i = 0; i < BLOCK; i++) sum += h[i];
    writeStr(h, 148, padOctal(sum, 7) + ' ', 8);      // 6 octal + NUL + space
    return h;
}

/**
 * Build a tar archive from a list of `{name, data}` entries.
 *
 * @param {Array<{name: string, data: Uint8Array | string}>} files
 * @returns {Uint8Array}
 */
export function tar(files) {
    const enc = new TextEncoder();
    const blocks = [];
    for (const f of files) {
        const data = typeof f.data === 'string' ? enc.encode(f.data) : f.data;
        blocks.push(buildHeader({ name: f.name, size: data.length }));
        blocks.push(data);
        const pad = (BLOCK - (data.length % BLOCK)) % BLOCK;
        if (pad) blocks.push(new Uint8Array(pad));
    }
    // Two zeroed blocks terminate the archive.
    blocks.push(new Uint8Array(BLOCK));
    blocks.push(new Uint8Array(BLOCK));

    const total = blocks.reduce((n, b) => n + b.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const b of blocks) { out.set(b, off); off += b.length; }
    return out;
}

/**
 * Gzip a byte buffer using the browser-native CompressionStream.
 * Available in modern Chromium/Firefox/Safari.
 */
export async function gzip(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Build and gzip a tar archive in one call.
 */
export async function tarGz(files) {
    return gzip(tar(files));
}
