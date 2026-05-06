#!/usr/bin/env node
// Read a CallVariantsOutput TFRecord shard, extract the
// `genotype_probabilities` (repeated double, field 3) for each record,
// and dump them as a JSON array. Used by the browser test as the
// native-ORT reference to diff in-browser predictions against.
//
// Usage:  extract-expected.mjs <cvos.tfrecord.gz> <expected.json>

import { readFileSync, writeFileSync } from 'node:fs';
import zlib from 'node:zlib';

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
    console.error('usage: extract-expected.mjs <cvos.tfrecord.gz> <expected.json>');
    process.exit(2);
}

function readVarint(buf, off) {
    let result = 0, shift = 0, i = off;
    while (true) {
        const b = buf[i++];
        result |= (b & 0x7f) << shift;
        if ((b & 0x80) === 0) return [result, i];
        shift += 7;
    }
}

function readTfRecords(gzPath) {
    const raw = zlib.gunzipSync(readFileSync(gzPath));
    const records = [];
    let off = 0;
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    while (off + 12 < raw.length) {
        const len = Number(view.getBigUint64(off, true));
        off += 8 + 4;
        records.push(raw.subarray(off, off + len));
        off += len + 4;
    }
    return records;
}

function extractGenotypeProbs(payload) {
    const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    let off = 0;
    while (off < payload.length) {
        const tag = payload[off++];
        const fieldNum = tag >> 3;
        const wireType = tag & 7;
        if (fieldNum === 3 && wireType === 2) {
            const [len, after] = readVarint(payload, off);
            off = after;
            const out = [];
            for (let i = 0; i < len; i += 8) out.push(view.getFloat64(off + i, true));
            return out;
        }
        if (wireType === 0) { const [, n] = readVarint(payload, off); off = n; }
        else if (wireType === 2) { const [len, after] = readVarint(payload, off); off = after + len; }
        else if (wireType === 1) off += 8;
        else if (wireType === 5) off += 4;
        else throw new Error(`unsupported wire type ${wireType}`);
    }
    return null;
}

const records = readTfRecords(inPath);
const probs = records.map(extractGenotypeProbs);
writeFileSync(outPath, JSON.stringify({ genotype_probabilities: probs }, null, 2));
console.log(`wrote ${probs.length} probability vectors to ${outPath}`);
