'use strict';

const fs = require('fs');
const path = require('path');
const ui = require('./ui');

/**
 * MCP server entries to install.
 *
 * Only include servers that are published to a public registry (PyPI or npm).
 * Unpublished packages (video-explainer-mcp, video-agent-mcp) are excluded
 * because `uvx` cannot resolve them — users who need them must add local
 * `uv run --directory` entries manually.
 */
const MCP_SERVERS = {
  'video-research': {
    command: 'uvx',
    args: ['--refresh', 'video-research-mcp[tracing]'],
  },
  playwright: {
    command: 'npx',
    args: ['@playwright/mcp@0.0.83', '--headless', '--caps=vision,pdf'],
  },
  'mlflow-mcp': {
    command: 'uvx',
    args: ['--with', 'mlflow[mcp]>=3.16.1,<4', 'mlflow', 'mcp', 'run'],
  },
};

/**
 * Servers previously installed by older versions that should be removed on
 * upgrade. These packages were never published to PyPI, so their entries
 * always fail. Cleaned up automatically during mergeConfig().
 */
const DEPRECATED_SERVERS = ['video-explainer', 'video-agent'];

/**
 * Return the path to the MCP config file.
 * Global: ~/.claude.json (Claude Code user scope)
 * Local:  ./.mcp.json (project root, not inside .claude/)
 */
function getConfigPath(mode) {
  if (mode === 'global') {
    const home = process.env.HOME || process.env.USERPROFILE;
    if (!home) {
      throw new Error(
        'Cannot determine home directory: HOME and USERPROFILE are both unset',
      );
    }
    return path.join(home, '.claude.json');
  }
  return path.join(process.cwd(), '.mcp.json');
}

/** Read and parse a JSON config file. Returns null if not found. Throws on malformed JSON. */
function readConfig(configPath) {
  try {
    const raw = fs.readFileSync(configPath, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    if (err instanceof SyntaxError) {
      ui.error(`Malformed JSON in ${configPath}`);
      ui.info('Fix the file manually or delete it to start fresh');
      throw err;
    }
    throw err;
  }
}

/** Deep-equal two MCP server entries (command + args). */
function entriesEqual(a, b) {
  if (!a || !b) return false;
  if (a.command !== b.command) return false;
  const aArgs = a.args || [];
  const bArgs = b.args || [];
  if (aArgs.length !== bArgs.length) return false;
  return aArgs.every((v, i) => v === bArgs[i]);
}

/**
 * Merge MCP_SERVERS into the config file.
 *
 * Existing user-configured entries are preserved by default: the installer
 * only writes a server entry when it is absent or matches the planned values
 * exactly. Pass ``forceOverwrite: true`` (mapped to the ``--force-mcp-overwrite``
 * CLI flag) to reset ``command`` and ``args`` of known entries to the installer
 * defaults; other keys on the entry (``env`` etc.) are kept.
 *
 * Returns ``{ writes, skipsMatching, skipsConflicting }`` so the caller can
 * report each outcome distinctly.
 *
 * @param {string} configPath
 * @param {{ forceOverwrite?: boolean, dryRun?: boolean }} [options]
 */
function mergeConfig(configPath, options = {}) {
  const { forceOverwrite = false, dryRun = false } = options;
  const existing = readConfig(configPath) || {};
  existing.mcpServers = existing.mcpServers || {};

  const writes = [];
  const skipsMatching = [];
  const skipsConflicting = [];

  for (const [name, config] of Object.entries(MCP_SERVERS)) {
    const current = existing.mcpServers[name];
    if (!current) {
      existing.mcpServers[name] = config;
      writes.push(name);
    } else if (forceOverwrite) {
      // Installer-managed keys (command, args) win; any other keys the user
      // added to the entry (env, etc.) are kept.
      existing.mcpServers[name] = { ...current, ...config };
      writes.push(name);
    } else if (entriesEqual(current, config)) {
      skipsMatching.push(name);
    } else {
      skipsConflicting.push({ name, current, planned: config });
    }
  }

  // Remove deprecated servers left by older installer versions
  for (const name of DEPRECATED_SERVERS) {
    const entry = existing.mcpServers[name];
    if (entry?.command === 'uvx' && entry.args?.length === 1 &&
        entry.args[0].startsWith(`${name}-mcp`)) {
      delete existing.mcpServers[name];
    }
  }

  if (!dryRun) {
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    fs.writeFileSync(configPath, JSON.stringify(existing, null, 2) + '\n', { mode: 0o600 });
  }
  return { writes, skipsMatching, skipsConflicting };
}

/** Remove MCP_SERVERS entries from the config file. Returns true if any were removed. */
function removeFromConfig(configPath) {
  const existing = readConfig(configPath);
  if (!existing?.mcpServers) return false;

  let removed = false;
  for (const name of Object.keys(MCP_SERVERS)) {
    if (JSON.stringify(existing.mcpServers[name]) === JSON.stringify(MCP_SERVERS[name])) {
      delete existing.mcpServers[name];
      removed = true;
    }
  }

  if (removed) {
    fs.writeFileSync(configPath, JSON.stringify(existing, null, 2) + '\n');
  }
  return removed;
}

/**
 * Template for ~/.config/video-research-mcp/.env.
 * Keys listed here will be appended (commented) if missing from an existing file.
 */
const ENV_TEMPLATE_KEYS = [
  'GEMINI_API_KEY',
  'GEMINI_MODEL',
  'GEMINI_FLASH_MODEL',
  'GEMINI_THINKING_LEVEL',
  'DEEP_RESEARCH_AGENT',
  'YOUTUBE_API_KEY',
  'WEAVIATE_URL',
  'WEAVIATE_API_KEY',
  'WEAVIATE_GRPC_URL',
  'MLFLOW_TRACKING_URI',
  'MLFLOW_EXPERIMENT_NAME',
  'EXPLAINER_PATH',
  'EXPLAINER_TTS_PROVIDER',
  'ELEVENLABS_API_KEY',
  'OPENAI_API_KEY',
  'WEAVIATE_VECTORIZER',
  'WEAVIATE_AUTO_MIGRATE',
  'LOCAL_FILE_ACCESS_ROOTS',
];

/**
 * Ensure ~/.config/video-research-mcp/.env exists with a commented template.
 * If the file already exists, append any new keys that are missing.
 * Returns the path to the env file.
 */
function ensureEnvFile() {
  const home = process.env.HOME || process.env.USERPROFILE;
  if (!home) return null;

  const envDir = path.join(home, '.config', 'video-research-mcp');
  const envPath = path.join(envDir, '.env');

  fs.mkdirSync(envDir, { recursive: true });

  if (!fs.existsSync(envPath)) {
    const lines = [
      '# video-research-mcp shared configuration',
      '# ─────────────────────────────────────────────────────────────',
      '# This file is read by the MCP server at startup.',
      '# Credentials are stored locally and sent to their configured providers.',
      '# Process env vars always take precedence over values here.',
      '#',
      '# Security:',
      '#   - This file is stored in your user config dir (chmod 600 recommended)',
      '#   - Never commit this file or share its values',
      '#   - Analysis requests send your selected content to the provider',
      '# ─────────────────────────────────────────────────────────────',
      '',
      '# Required — get yours at https://aistudio.google.com/apikey',
      '# GEMINI_API_KEY=',
      '# GEMINI_MODEL=gemini-3.8-flash',
      '# GEMINI_FLASH_MODEL=gemini-3.8-flash',
      '# GEMINI_THINKING_LEVEL=medium',
      '# DEEP_RESEARCH_AGENT=',
      '',
      '# Optional — falls back to GEMINI_API_KEY if not set',
      '# YOUTUBE_API_KEY=',
      '',
      '# Optional — knowledge store (leave commented to disable)',
      '# WEAVIATE_URL=',
      '# WEAVIATE_API_KEY=',
      '# WEAVIATE_GRPC_URL=',
      '#',
      '# Vectorizer: auto-detects based on OPENAI_API_KEY.',
      '#   weaviate = built-in embeddings, no extra key (good for Docker)',
      '#   openai   = text2vec-openai, requires OPENAI_API_KEY',
      '#   ollama   = local embeddings; configure Ollama and its model first',
      '# WEAVIATE_VECTORIZER=',
      '# WEAVIATE_AUTO_MIGRATE=',
      '',
      '# Optional — MLflow tracing (leave commented to disable)',
      '# MLFLOW_TRACKING_URI=http://127.0.0.1:5001',
      '# MLFLOW_EXPERIMENT_NAME=video-research-mcp',
      '',
      '# Video Explainer — path to cloned video_explainer repo (required for /ve: commands)',
      '# EXPLAINER_PATH=',
      '# EXPLAINER_TTS_PROVIDER=mock',
      '',
      '# Optional — ElevenLabs TTS (recommended for production)',
      '# ELEVENLABS_API_KEY=',
      '',
      '# Optional — OpenAI TTS (budget alternative)',
      '# OPENAI_API_KEY=',
      '',
      '# Strongly recommended: restrict which directories the server can read.',
      '# Tools that ingest local files accept paths supplied by the LLM, so',
      '# untrusted content (web pages, transcripts) can induce reads of files',
      '# you did not intend. Configure an OS-pathsep-separated allow-list and',
      '# the server will reject paths outside it. See README "Local file access',
      '# boundaries" for details.',
      '# LOCAL_FILE_ACCESS_ROOTS=' + path.join('${HOME}', 'dev') + path.delimiter + path.join('${HOME}', 'Downloads'),
      '',
    ];
    fs.writeFileSync(envPath, lines.join('\n'), { mode: 0o600 });
    return { path: envPath, created: true, added: 0 };
  }

  // Append missing keys
  const existing = fs.readFileSync(envPath, 'utf8');
  const missing = ENV_TEMPLATE_KEYS.filter(
    (k) => !existing.includes(`${k}=`),
  );
  if (missing.length > 0) {
    const suffix =
      '\n# Added by installer\n' + missing.map((k) => `# ${k}=`).join('\n') + '\n';
    fs.appendFileSync(envPath, suffix);
  }
  return { path: envPath, created: false, added: missing.length };
}

module.exports = {
  MCP_SERVERS,
  DEPRECATED_SERVERS,
  getConfigPath,
  readConfig,
  mergeConfig,
  removeFromConfig,
  ensureEnvFile,
  entriesEqual,
};
