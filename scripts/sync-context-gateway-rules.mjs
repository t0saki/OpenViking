#!/usr/bin/env node
// The plugin data module is authoritative; the Python wheel carries its JSON projection.
import { readFile, writeFile } from 'node:fs/promises';
import { CLIENT_RULES } from '../examples/memory-plugin-shared/lib/client-rules.mjs';

const target = new URL('../openviking_context_gateway/client-rules.json', import.meta.url);
const expected = `${JSON.stringify(CLIENT_RULES, null, 2)}\n`;
if (process.argv.includes('--check')) {
  if (await readFile(target, 'utf8') !== expected) {
    throw new Error('Run node scripts/sync-context-gateway-rules.mjs');
  }
} else {
  await writeFile(target, expected);
}
