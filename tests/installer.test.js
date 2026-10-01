'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const config = require('../bin/lib/config');
const { readManifest, computeActions, hashFile } = require('../bin/lib/manifest');

function fixture(t) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'gr-installer-'));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  return home;
}

function runInstaller(home, ...args) {
  return spawnSync(process.execPath, [path.resolve(__dirname, '../bin/install.js'), ...args], {
    cwd: home,
    env: { ...process.env, HOME: home, USERPROFILE: home },
    encoding: 'utf8',
  });
}

test('global registration uses Claude user scope and preserves user configuration', (t) => {
  const home = fixture(t);
  const configPath = path.join(home, '.claude.json');
  fs.writeFileSync(configPath, JSON.stringify({
    projects: { '/other/project': { hasTrustDialogAccepted: true } },
    mcpServers: {
      personal: { command: 'my-server' },
      'video-research': { command: 'uvx', args: ['old'], env: { GEMINI_MODEL: 'custom' } },
      'video-agent': { command: 'uv', args: ['run', 'video-agent-mcp'] },
    },
  }));
  const result = runInstaller(home, '--global');
  assert.equal(result.status, 0, result.stderr);
  const installed = JSON.parse(fs.readFileSync(configPath));
  assert.equal(installed.projects['/other/project'].hasTrustDialogAccepted, true);
  assert.deepEqual(installed.mcpServers.personal, { command: 'my-server' });
  assert.equal(installed.mcpServers['video-research'].env.GEMINI_MODEL, 'custom');
  // A customized command/args entry is preserved by default (see
  // tests/installer/test_config_merge.js); --force-mcp-overwrite resets it.
  assert.deepEqual(installed.mcpServers['video-research'].args, ['old']);
  assert.match(result.stdout + result.stderr, /Preserved existing 'video-research' entry/);
  assert.equal(installed.mcpServers['video-agent'].command, 'uv');
  assert.equal(fs.existsSync(path.join(home, '.claude/.mcp.json')), false);
});

test('force-mcp-overwrite resets installer args and keeps user env', (t) => {
  // Guards: the escape hatch must refresh stale command/args on upgrade
  // without discarding keys the user added to the entry.
  const home = fixture(t);
  const configPath = path.join(home, '.claude.json');
  fs.writeFileSync(configPath, JSON.stringify({ mcpServers: {
    'video-research': { command: 'uvx', args: ['old'], env: { GEMINI_MODEL: 'custom' } },
  } }));
  const result = runInstaller(home, '--global', '--force-mcp-overwrite');
  assert.equal(result.status, 0, result.stderr);
  const installed = JSON.parse(fs.readFileSync(configPath));
  assert.equal(installed.mcpServers['video-research'].env.GEMINI_MODEL, 'custom');
  assert.deepEqual(installed.mcpServers['video-research'].args, config.MCP_SERVERS['video-research'].args);
});

test('local install registers at the project root and creates private credential template', (t) => {
  const home = fixture(t);
  const result = runInstaller(home, '--local');
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(path.join(home, '.mcp.json')), true);
  const envPath = path.join(home, '.config/video-research-mcp/.env');
  assert.match(fs.readFileSync(envPath, 'utf8'), /GEMINI_MODEL=gemini-3\.8-flash/);
  if (process.platform !== 'win32') assert.equal(fs.statSync(envPath).mode & 0o777, 0o600);
});

test('upgrade and uninstall retain modified files and their ownership evidence', (t) => {
  const home = fixture(t);
  assert.equal(runInstaller(home, '--global').status, 0);
  const target = path.join(home, '.claude');
  const modified = path.join(target, 'commands/gr/video.md');
  fs.appendFileSync(modified, '\nUser addition\n');
  const before = readManifest(target).files['commands/gr/video.md'].hash;
  assert.equal(runInstaller(home, '--global').status, 0);
  assert.equal(readManifest(target).files['commands/gr/video.md'].hash, before);
  assert.equal(runInstaller(home, '--global', '--uninstall').status, 0);
  assert.match(fs.readFileSync(modified, 'utf8'), /User addition/);
  assert.equal(readManifest(target).files['commands/gr/video.md'].hash, before);
});

test('a missing manifest cannot authorize overwriting or adopting an existing file', (t) => {
  const home = fixture(t);
  const userFile = path.join(home, '.claude/commands/gr/video.md');
  fs.mkdirSync(path.dirname(userFile), { recursive: true });
  fs.writeFileSync(userFile, 'Existing user command');
  assert.equal(runInstaller(home, '--global').status, 0);
  assert.equal(fs.readFileSync(userFile, 'utf8'), 'Existing user command');
  assert.equal(readManifest(path.join(home, '.claude')).files['commands/gr/video.md'], undefined);
  assert.equal(runInstaller(home, '--global', '--uninstall').status, 0);
  assert.equal(fs.readFileSync(userFile, 'utf8'), 'Existing user command');
});

test('force explicitly replaces a protected user edit', (t) => {
  const home = fixture(t);
  assert.equal(runInstaller(home, '--local').status, 0);
  const userFile = path.join(home, '.claude/commands/gr/video.md');
  fs.writeFileSync(userFile, 'User customization');
  assert.equal(runInstaller(home, '--local', '--force').status, 0);
  assert.notEqual(fs.readFileSync(userFile, 'utf8'), 'User customization');
});

test('malformed manifests fail before changing files', (t) => {
  const home = fixture(t);
  const target = path.join(home, '.claude');
  fs.mkdirSync(target);
  fs.writeFileSync(path.join(target, 'gr-file-manifest.json'), '{broken');
  const result = runInstaller(home, '--global');
  assert.notEqual(result.status, 0);
  assert.equal(fs.existsSync(path.join(target, 'commands')), false);
});

test('manifest traversal fails before uninstall can delete outside the install tree', (t) => {
  const home = fixture(t);
  const outside = path.join(home, 'keep.txt');
  fs.writeFileSync(outside, 'Keep');
  const target = path.join(home, '.claude');
  fs.mkdirSync(target);
  fs.writeFileSync(path.join(target, 'gr-file-manifest.json'), JSON.stringify({
    installedAt: 'test', files: { '../keep.txt': { hash: hashFile(outside) } },
  }));
  const result = runInstaller(home, '--global', '--uninstall');
  assert.notEqual(result.status, 0);
  assert.equal(fs.readFileSync(outside, 'utf8'), 'Keep');
});

test('a namespace prefix cannot authorize deleting unrelated Claude settings', (t) => {
  const home = fixture(t);
  const target = path.join(home, '.claude');
  fs.mkdirSync(target);
  const settings = path.join(target, 'settings.json');
  fs.writeFileSync(settings, 'User settings');
  fs.writeFileSync(path.join(target, 'gr-file-manifest.json'), JSON.stringify({
    installedAt: 'test', files: { 'commands/../settings.json': { hash: hashFile(settings) } },
  }));
  assert.notEqual(runInstaller(home, '--global', '--uninstall').status, 0);
  assert.equal(fs.readFileSync(settings, 'utf8'), 'User settings');
});

test('symlinked managed directories cannot redirect install or uninstall writes', (t) => {
  const home = fixture(t);
  const target = path.join(home, '.claude');
  const outside = path.join(home, 'outside');
  fs.mkdirSync(target);
  fs.mkdirSync(outside);
  fs.symlinkSync(outside, path.join(target, 'commands'), 'dir');
  const keep = path.join(outside, 'keep.md');
  fs.writeFileSync(keep, 'Keep');
  assert.notEqual(runInstaller(home, '--global').status, 0);
  fs.writeFileSync(path.join(target, 'gr-file-manifest.json'), JSON.stringify({
    installedAt: 'test', files: { 'commands/keep.md': { hash: hashFile(keep) } },
  }));
  assert.notEqual(runInstaller(home, '--global', '--uninstall').status, 0);
  assert.equal(fs.readFileSync(keep, 'utf8'), 'Keep');
});

test('a manifest entry without ownership hash cannot authorize removal', (t) => {
  const home = fixture(t);
  assert.throws(() => computeActions(home, home, {}, { files: { 'commands/user.md': {} } }, false),
    /Invalid install manifest hash/);
});

test('modified obsolete files remain protected across upgrades', (t) => {
  const home = fixture(t);
  assert.equal(runInstaller(home, '--global').status, 0);
  const target = path.join(home, '.claude');
  const obsolete = path.join(target, 'commands/gr/old.md');
  fs.writeFileSync(obsolete, 'Old command');
  const manifestPath = path.join(target, 'gr-file-manifest.json');
  const manifest = readManifest(target);
  manifest.files['commands/gr/old.md'] = { hash: hashFile(obsolete) };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  fs.appendFileSync(obsolete, ' user edit');
  assert.equal(runInstaller(home, '--global').status, 0);
  assert.equal(readManifest(target).files['commands/gr/old.md'].hash,
    manifest.files['commands/gr/old.md'].hash);
  assert.equal(runInstaller(home, '--global', '--uninstall').status, 0);
  assert.equal(fs.readFileSync(obsolete, 'utf8'), 'Old command user edit');
});

test('uninstall preserves a customized MCP server', (t) => {
  const home = fixture(t);
  const configPath = path.join(home, '.claude.json');
  fs.writeFileSync(configPath, JSON.stringify({ mcpServers: {
    'video-research': { ...config.MCP_SERVERS['video-research'], env: { GEMINI_MODEL: 'custom' } },
  } }));
  assert.equal(config.removeFromConfig(configPath), false);
  assert.equal(JSON.parse(fs.readFileSync(configPath)).mcpServers['video-research'].env.GEMINI_MODEL, 'custom');
});

test('action planning rejects unsafe paths even without reading a manifest file', (t) => {
  const home = fixture(t);
  assert.throws(() => computeActions(home, home, {}, { files: { '/outside': {} } }, false),
    /Unsafe install manifest path/);
});

// ---------------------------------------------------------------------------
// mergeConfig(): preserve user-customized MCP entries unless forced.
// Each test names the regression it guards against.
// ---------------------------------------------------------------------------

function mergeFixture(t) {
  return path.join(fixture(t), '.mcp.json');
}

function writeJSON(p, data) {
  fs.writeFileSync(p, JSON.stringify(data, null, 2) + '\n');
}

test('mergeConfig fresh install writes every MCP_SERVERS entry', (t) => {
  // Guards: the installer failing to seed entries on a clean machine.
  const cfgPath = mergeFixture(t);
  const result = config.mergeConfig(cfgPath);
  const expectedNames = Object.keys(config.MCP_SERVERS).sort();
  assert.deepEqual(result.writes.sort(), expectedNames);
  assert.deepEqual(result.skipsMatching, []);
  assert.deepEqual(result.skipsConflicting, []);
  const onDisk = config.readConfig(cfgPath);
  for (const name of expectedNames) {
    assert.deepEqual(onDisk.mcpServers[name], config.MCP_SERVERS[name]);
  }
});

test('mergeConfig skips exact-match entries silently', (t) => {
  // Guards: noisy "updated" output when nothing changed.
  const cfgPath = mergeFixture(t);
  writeJSON(cfgPath, { mcpServers: { ...config.MCP_SERVERS } });
  const result = config.mergeConfig(cfgPath);
  assert.deepEqual(result.writes, []);
  assert.deepEqual(result.skipsMatching.sort(), Object.keys(config.MCP_SERVERS).sort());
  assert.deepEqual(result.skipsConflicting, []);
});

test('mergeConfig preserves a user-customized playwright entry by default', (t) => {
  // Guards: the installer silently replacing a user's pinned Playwright MCP
  // version on every upgrade.
  const cfgPath = mergeFixture(t);
  const userEntry = { command: 'npx', args: ['@playwright/mcp@1.2.3', '--headless', '--caps=vision'] };
  writeJSON(cfgPath, { mcpServers: { playwright: userEntry } });
  const result = config.mergeConfig(cfgPath);
  assert.equal(result.skipsConflicting.length, 1);
  assert.equal(result.skipsConflicting[0].name, 'playwright');
  assert.deepEqual(config.readConfig(cfgPath).mcpServers.playwright, userEntry);
});

test('mergeConfig forceOverwrite replaces user command and args', (t) => {
  // Guards: the --force-mcp-overwrite escape hatch must still apply defaults.
  const cfgPath = mergeFixture(t);
  writeJSON(cfgPath, { mcpServers: { playwright: { command: 'npx', args: ['@playwright/mcp@1.2.3'] } } });
  const result = config.mergeConfig(cfgPath, { forceOverwrite: true });
  assert.ok(result.writes.includes('playwright'));
  assert.deepEqual(config.readConfig(cfgPath).mcpServers.playwright, config.MCP_SERVERS.playwright);
});

test('mergeConfig dryRun reports planned actions without writing', (t) => {
  // Guards: --check mutating the config of a user who only wanted to inspect it.
  const cfgPath = mergeFixture(t);
  const before = JSON.stringify({ mcpServers: { existing: { command: 'x' } } });
  fs.writeFileSync(cfgPath, before);
  const result = config.mergeConfig(cfgPath, { dryRun: true });
  assert.ok(result.writes.length > 0);
  assert.equal(fs.readFileSync(cfgPath, 'utf8'), before);
});

test('mergeConfig writes one entry, preserves another and no-ops a third', (t) => {
  // Guards: a real upgrade with one customized, one matching and one missing
  // entry must report all three outcomes correctly in a single call.
  const cfgPath = mergeFixture(t);
  const customizedPlaywright = { command: 'npx', args: ['@playwright/mcp@9.9.9'] };
  writeJSON(cfgPath, { mcpServers: {
    playwright: customizedPlaywright,
    'video-research': { ...config.MCP_SERVERS['video-research'] },
  } });
  const result = config.mergeConfig(cfgPath);
  assert.deepEqual(result.writes, ['mlflow-mcp']);
  assert.deepEqual(result.skipsMatching, ['video-research']);
  assert.equal(result.skipsConflicting.length, 1);
  assert.equal(result.skipsConflicting[0].name, 'playwright');
  assert.deepEqual(config.readConfig(cfgPath).mcpServers.playwright, customizedPlaywright);
});

test('mergeConfig leaves servers it does not own untouched', (t) => {
  // Guards: a user's own MCP server being modified by any merge path.
  const cfgPath = mergeFixture(t);
  const foreign = { command: 'node', args: ['my-custom-mcp.js'] };
  writeJSON(cfgPath, { mcpServers: { 'my-custom': foreign } });
  config.mergeConfig(cfgPath);
  assert.deepEqual(config.readConfig(cfgPath).mcpServers['my-custom'], foreign);
});

test('entriesEqual compares command and args only, and guards nulls', () => {
  // Guards: a false match silently overwriting a differing entry, or a false
  // mismatch rewriting an identical entry on every install.
  const a = { command: 'uvx', args: ['x', 'y'] };
  assert.equal(config.entriesEqual(a, { command: 'uvx', args: ['x', 'y'] }), true);
  assert.equal(config.entriesEqual(a, { command: 'pipx', args: ['x', 'y'] }), false);
  assert.equal(config.entriesEqual(a, { command: 'uvx', args: ['x', 'z'] }), false);
  assert.equal(config.entriesEqual(a, { command: 'uvx', args: ['x'] }), false);
  assert.equal(config.entriesEqual(null, a), false);
  assert.equal(config.entriesEqual(a, undefined), false);
});
