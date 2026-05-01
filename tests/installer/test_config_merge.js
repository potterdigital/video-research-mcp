'use strict';

/**
 * Tests for bin/lib/config.js mergeConfig() — preserve user-customised
 * MCP entries unless --force-mcp-overwrite is passed.
 *
 * Run: node tests/installer/test_config_merge.js
 *
 * Plain-Node test runner (no devDeps) — exits 0 on success, non-zero on
 * the first failure. Each test names the regression it guards against.
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { MCP_SERVERS, mergeConfig, readConfig, entriesEqual } = require(
  '../../bin/lib/config',
);

let passed = 0;
let failed = 0;

function test(name, fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vrm-cfg-'));
  const cfgPath = path.join(dir, '.mcp.json');
  try {
    fn(cfgPath);
    process.stderr.write(`  ✓ ${name}\n`);
    passed++;
  } catch (err) {
    process.stderr.write(`  ✗ ${name}\n    ${err.message}\n`);
    failed++;
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function writeJSON(p, data) {
  fs.writeFileSync(p, JSON.stringify(data, null, 2) + '\n');
}

// ---------------------------------------------------------------------------

test(
  'fresh install: writes all MCP_SERVERS entries when config is absent',
  // Guards: regression where the installer fails to seed entries on a clean machine.
  (cfgPath) => {
    const result = mergeConfig(cfgPath);
    const expectedNames = Object.keys(MCP_SERVERS).sort();
    assert.deepEqual(result.writes.sort(), expectedNames);
    assert.deepEqual(result.skipsMatching, []);
    assert.deepEqual(result.skipsConflicting, []);
    const onDisk = readConfig(cfgPath);
    for (const name of expectedNames) {
      assert.deepEqual(onDisk.mcpServers[name], MCP_SERVERS[name]);
    }
  },
);

test(
  'idempotent: existing exact-match entries are skipped silently',
  // Guards: noisy "updated" output when nothing changed.
  (cfgPath) => {
    writeJSON(cfgPath, { mcpServers: { ...MCP_SERVERS } });
    const result = mergeConfig(cfgPath);
    assert.deepEqual(result.writes, []);
    assert.deepEqual(
      result.skipsMatching.sort(),
      Object.keys(MCP_SERVERS).sort(),
    );
    assert.deepEqual(result.skipsConflicting, []);
  },
);

test(
  'preserves user-customised playwright entry on default install',
  // Guards: the CRITICAL bug — installer silently downgraded a user's
  // pinned Playwright MCP version on every upgrade.
  (cfgPath) => {
    const userEntry = {
      command: 'npx',
      args: ['@playwright/mcp@1.2.3', '--headless', '--caps=vision'],
    };
    writeJSON(cfgPath, { mcpServers: { playwright: userEntry } });

    const result = mergeConfig(cfgPath);
    assert.equal(result.skipsConflicting.length, 1);
    assert.equal(result.skipsConflicting[0].name, 'playwright');

    const onDisk = readConfig(cfgPath);
    assert.deepEqual(
      onDisk.mcpServers.playwright,
      userEntry,
      'User playwright entry must be preserved verbatim',
    );
  },
);

test(
  '--force-mcp-overwrite replaces user entries',
  // Guards: escape hatch must still work for users who explicitly want
  // the installer's defaults.
  (cfgPath) => {
    const userEntry = {
      command: 'npx',
      args: ['@playwright/mcp@1.2.3'],
    };
    writeJSON(cfgPath, { mcpServers: { playwright: userEntry } });

    const result = mergeConfig(cfgPath, { forceOverwrite: true });
    assert.ok(result.writes.includes('playwright'));
    const onDisk = readConfig(cfgPath);
    assert.deepEqual(onDisk.mcpServers.playwright, MCP_SERVERS.playwright);
  },
);

test(
  'dryRun reports planned actions without writing the file',
  // Guards: --check mode must never mutate config (would surprise users
  // running it just to inspect state).
  (cfgPath) => {
    const before = JSON.stringify({ mcpServers: { existing: { command: 'x' } } });
    fs.writeFileSync(cfgPath, before);

    const result = mergeConfig(cfgPath, { dryRun: true });
    assert.ok(result.writes.length > 0);
    assert.equal(fs.readFileSync(cfgPath, 'utf8'), before);
  },
);

test(
  'mixed: writes one entry, preserves another, no-ops a third',
  // Guards: real-world upgrade where a user has customised one entry,
  // matches another exactly, and is missing the third. All three paths
  // must be reported correctly in a single call.
  (cfgPath) => {
    const customisedPlaywright = {
      command: 'npx',
      args: ['@playwright/mcp@9.9.9'],
    };
    writeJSON(cfgPath, {
      mcpServers: {
        playwright: customisedPlaywright,
        'video-research': { ...MCP_SERVERS['video-research'] },
        // mlflow-mcp absent
      },
    });

    const result = mergeConfig(cfgPath);
    assert.deepEqual(result.writes, ['mlflow-mcp']);
    assert.deepEqual(result.skipsMatching, ['video-research']);
    assert.equal(result.skipsConflicting.length, 1);
    assert.equal(result.skipsConflicting[0].name, 'playwright');

    const onDisk = readConfig(cfgPath);
    assert.deepEqual(onDisk.mcpServers.playwright, customisedPlaywright);
  },
);

test(
  'preserves unrelated user entries (foreign servers)',
  // Guards: an MCP server name the installer does not own (e.g. user's
  // own custom MCP) must be untouched in every code path.
  (cfgPath) => {
    const foreign = { command: 'node', args: ['my-custom-mcp.js'] };
    writeJSON(cfgPath, { mcpServers: { 'my-custom': foreign } });

    mergeConfig(cfgPath);
    const onDisk = readConfig(cfgPath);
    assert.deepEqual(onDisk.mcpServers['my-custom'], foreign);
  },
);

test(
  'entriesEqual: matches identical, differs on command, differs on args',
  // Guards: false-positive matches would silently overwrite differing
  // user entries; false-negative matches would noisy-rewrite identical
  // entries on every install.
  () => {
    const a = { command: 'uvx', args: ['x', 'y'] };
    const b = { command: 'uvx', args: ['x', 'y'] };
    const c = { command: 'pipx', args: ['x', 'y'] };
    const d = { command: 'uvx', args: ['x', 'z'] };
    const e = { command: 'uvx', args: ['x'] };

    assert.equal(entriesEqual(a, b), true);
    assert.equal(entriesEqual(a, c), false);
    assert.equal(entriesEqual(a, d), false);
    assert.equal(entriesEqual(a, e), false);
    assert.equal(entriesEqual(null, a), false);
    assert.equal(entriesEqual(a, undefined), false);
  },
);

// ---------------------------------------------------------------------------

process.stderr.write(`\nResults: ${passed} passed, ${failed} failed\n`);
process.exit(failed === 0 ? 0 : 1);
