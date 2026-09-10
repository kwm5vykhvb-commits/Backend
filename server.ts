import express, { Request, Response } from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { TelegramClient } from "telegram";
import { StringSession } from "telegram/sessions/index.js";
import bigInt from "big-integer";
import { sanitizeFileName, filterAndSortEpisodes, ParsedMedia } from "./cleaner.js";
import {
  isGeminiAvailable,
  aiDeduplicateAndSortEpisodes,
  aiIdentifyAnimeFromPoster,
  aiSmartSearchAssistant,
  aiChatAssistant,
  aiEnrichAnimeMetadata,
  getActiveGeminiModel,
  DeduplicatedEpisode,
  PosterIdentificationResult,
  SmartSearchResult,
  ChatMessage,
  EnrichedMetadataResult,
} from "./gemini.js";

dotenv.config();

const app = express();

// Configuration du port :
// - Sur Render, Render injecte RENDER=true et la variable PORT (par défaut 10000).
// - Dans le conteneur AI Studio (où le proxy Nginx écoute sur 8080 et redirige vers 3000), le port 3000 est requis.
const isRender = process.env.RENDER === "true" || !!process.env.RENDER;
const PORT = isRender || (process.env.PORT && process.env.PORT !== "8080")
  ? parseInt(process.env.PORT || "10000", 10)
  : 3000;

app.use(cors());
app.use(express.json({ limit: "25mb" }));

// --- TELEGRAM CLIENT SINGLETON (GramJS MTProto) ---
export interface TelegramStatusInfo {
  connected: boolean;
  connecting: boolean;
  user: { id?: string; username?: string | null; firstName?: string } | null;
  error: {
    code?: number;
    type?: string;
    message: string;
    detail?: string;
    resolution?: string;
    timestamp: string;
  } | null;
  channelsCount: number;
  sessionConfigured: boolean;
}

let tgClient: TelegramClient | null = null;
let tgConnectingPromise: Promise<TelegramClient | null> | null = null;
let activeSessionString = process.env.SESSION_STRING || "";
let lastConnectionAttemptTime = 0;
const CONNECTION_COOLDOWN_MS = 20000; // 20s cooldown on fatal 406 errors to avoid hammering Telegram servers

export const tgStatusInfo: TelegramStatusInfo = {
  connected: false,
  connecting: false,
  user: null,
  error: null,
  channelsCount: 0,
  sessionConfigured: !!(process.env.SESSION_STRING && process.env.SESSION_STRING.length > 10),
};

export async function closeTelegramClient(): Promise<void> {
  if (tgClient) {
    try {
      console.log("[Telegram MTProto] Fermeture propre du client...");
      await tgClient.disconnect();
      await tgClient.destroy();
    } catch (e: any) {
      console.warn("[Telegram MTProto] Avertissement lors de la fermeture:", e?.message || e);
    } finally {
      tgClient = null;
      tgStatusInfo.connected = false;
      tgStatusInfo.connecting = false;
    }
  }
}

export async function setTelegramSession(newSessionStr: string): Promise<boolean> {
  activeSessionString = (newSessionStr || "").trim();
  tgStatusInfo.sessionConfigured = !!(activeSessionString && activeSessionString.length > 10);
  tgStatusInfo.error = null;
  lastConnectionAttemptTime = 0;
  await closeTelegramClient();
  const client = await getTelegramClient(true);
  return !!client;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, errorMsg: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error(errorMsg)), timeoutMs)),
  ]);
}

export async function getTelegramClient(forceReconnect = false): Promise<TelegramClient | null> {
  if (!forceReconnect && tgClient && tgClient.connected) {
    return tgClient;
  }
  if (tgConnectingPromise) {
    return tgConnectingPromise;
  }

  const apiId = parseInt(process.env.API_ID || "0", 10);
  const apiHash = process.env.API_HASH || "";
  const sessionStr = activeSessionString || process.env.SESSION_STRING || "";

  if (!apiId || !apiHash || !sessionStr) {
    tgStatusInfo.connected = false;
    tgStatusInfo.connecting = false;
    tgStatusInfo.error = {
      message: "Identifiants Telegram manquants dans l'environnement (API_ID, API_HASH, SESSION_STRING)",
      timestamp: new Date().toISOString(),
    };
    return null;
  }

  // Si une erreur 406 AUTH_KEY_DUPLICATED s'est produite récemment, respecter un temps de pause
  // sauf si une reconnexion forcée (ex: nouvelle session) est demandée
  if (!forceReconnect && tgStatusInfo.error?.code === 406) {
    const elapsed = Date.now() - lastConnectionAttemptTime;
    if (elapsed < CONNECTION_COOLDOWN_MS) {
      return null;
    }
  }

  tgStatusInfo.connecting = true;
  lastConnectionAttemptTime = Date.now();

  tgConnectingPromise = (async () => {
    let clientInstance: TelegramClient | null = null;
    try {
      const session = new StringSession(sessionStr);
      clientInstance = new TelegramClient(session, apiId, apiHash, {
        connectionRetries: 1,
        useWSS: false,
      });

      await withTimeout(clientInstance.connect(), 6000, "Timeout connexion Telegram MTProto (6s)");
      tgClient = clientInstance;
      tgStatusInfo.connected = true;
      tgStatusInfo.connecting = false;
      tgStatusInfo.error = null;

      try {
        const me = await clientInstance.getMe();
        tgStatusInfo.user = {
          id: me.id?.toString(),
          username: me.username || null,
          firstName: me.firstName || "",
        };
        console.log(`[Telegram MTProto] Connecté avec succès en tant que @${me.username || me.firstName} (${me.id})`);
      } catch (meErr: any) {
        console.log("[Telegram MTProto] Connecté avec succès");
      }

      return clientInstance;
    } catch (err: any) {
      const errMsg = err?.message || String(err);
      const isDuplicated = err?.code === 406 || errMsg.includes("406") || errMsg.includes("AUTH_KEY_DUPLICATED");

      if (clientInstance) {
        try {
          await clientInstance.destroy();
        } catch {}
      }
      tgClient = null;
      tgStatusInfo.connected = false;
      tgStatusInfo.connecting = false;

      if (isDuplicated) {
        tgStatusInfo.error = {
          code: 406,
          type: "AUTH_KEY_DUPLICATED",
          message: "Session Telegram dupliquée / révoquée (AUTH_KEY_DUPLICATED 406)",
          detail: "Cette session Telegram (SESSION_STRING) a été connectée simultanément depuis une autre instance (ex: Render et environnement de développement). Telegram invalide la clé dès qu'un conflit survient.",
          resolution: "Générez une nouvelle SESSION_STRING avec la commande 'npm run session' ou mettez à jour votre clé depuis l'interface.",
          timestamp: new Date().toISOString(),
        };
        console.warn("[Telegram MTProto] Conflit 406 AUTH_KEY_DUPLICATED : La SESSION_STRING doit être renouvelée ou dédiée à une seule instance active.");
      } else {
        tgStatusInfo.error = {
          code: err?.code,
          type: err?.errorMessage || "CONNECTION_FAILED",
          message: errMsg,
          timestamp: new Date().toISOString(),
        };
        console.error("Failed to connect to Telegram MTProto:", errMsg);
      }
      return null;
    } finally {
      tgConnectingPromise = null;
    }
  })();

  return tgConnectingPromise;
}

// --- ANTI-BAN / Cache logic matching main.py ---
const SEARCH_CACHE_TTL = 300; // 5 minutes in seconds
const RESULTS_PAGE_SIZE = 100;

export interface EpisodeItem extends ParsedMedia {
  message_id: number;
  title: string;
  file_name: string;
  size_mb: number;
  stream_url: string;
}

interface CacheEntry {
  data: {
    anime_info: AnimeMetadata;
    episodes: EpisodeItem[];
  };
  cached_at: number;
}

const _search_cache = new Map<string, CacheEntry>();

function getCachedSearch(cacheKey: string) {
  const cached = _search_cache.get(cacheKey);
  if (!cached) return null;
  const now = Math.floor(Date.now() / 1000);
  if (now - cached.cached_at > SEARCH_CACHE_TTL) {
    _search_cache.delete(cacheKey);
    return null;
  }
  return cached.data;
}

function setCachedSearch(cacheKey: string, data: { anime_info: AnimeMetadata; episodes: EpisodeItem[] }) {
  const now = Math.floor(Date.now() / 1000);
  // Opportunistically drop expired entries
  for (const [k, v] of _search_cache.entries()) {
    if (now - v.cached_at > SEARCH_CACHE_TTL) {
      _search_cache.delete(k);
    }
  }
  _search_cache.set(cacheKey, { data, cached_at: now });
}

// --- Episode number extraction patterns from main.py ---
const EPISODE_NUMBER_PATTERNS = [
  /s(?:eason)?\s*\d{1,2}[\s._-]*(?:e(?:p(?:isode)?)?[\s._-]*)?(\d{1,4})/i, // S01E05, Season 1 Episode 05, Season 3 - 05
  /(?<![A-Za-z0-9])(?:episode|ep)[\s._-]*(\d{1,4})(?:v\d+)?(?![A-Za-z0-9])/i, // Episode 05, Ep.05, Ep_05
  /(?<![A-Za-z0-9])e[\s._-]*(\d{1,4})(?:v\d+)?(?![A-Za-z0-9])/i, // E05
  /(?:^|[\[\(\s._-])(\d{1,4})(?:v\d+)?(?:[\]\)\s._-]|$)/, // standalone "- 05 -", "[05]", "(05)"
];

export function extractEpisodeNumber(fileName: string): number | null {
  for (const pattern of EPISODE_NUMBER_PATTERNS) {
    const match = pattern.exec(fileName);
    if (match && match[1]) {
      const num = parseInt(match[1], 10);
      if (!isNaN(num)) {
        return num;
      }
    }
  }
  return null;
}

interface AnimeMetadata {
  title: string;
  cover: string | null;
  banner: string | null;
  synopsis: string;
  score: string | null;
  genres: string[];
  total_episodes_official: number | null;
  year: number | null;
}

export function getScore(episode: { title?: string; file_name?: string }, query: string): number {
  const q = query.toLowerCase().trim();
  const title = (episode.title || "").toLowerCase().trim();
  const fileName = (episode.file_name || "").toLowerCase().trim();

  if (title === q) return 1000;
  if (title.startsWith(q)) return 900;
  if (title.includes(q)) return 700;
  if (fileName.includes(q)) return 600;
  return 0;
}

const POPULAR_ANIME_CATALOG: Record<string, Partial<AnimeMetadata>> = {
  naruto: {
    title: "Naruto",
    cover: "https://media.kitsu.app/anime/poster_images/11/large.jpg",
    banner: "https://media.kitsu.app/anime/cover_images/11/large.jpg",
    synopsis: "Naruto Uzumaki, un jeune ninja farceur du village caché de Konoha, rêve de devenir Hokage, le chef du village.",
    score: "82.5%",
    genres: ["Action", "Aventure", "Ninja"],
    total_episodes_official: 220,
    year: 2002,
  },
  "naruto shippuden": {
    title: "Naruto: Shippuuden",
    cover: "https://media.kitsu.app/anime/poster_images/1555/large.jpg",
    banner: "https://media.kitsu.app/anime/cover_images/1555/large.jpg",
    synopsis: "Deux ans et demi après son départ avec Jiraiya, Naruto revient à Konoha pour affronter l'organisation criminelle Akatsuki.",
    score: "84.1%",
    genres: ["Action", "Aventure", "Shonen"],
    total_episodes_official: 500,
    year: 2007,
  },
  "one piece": {
    title: "One Piece",
    cover: "https://media.kitsu.app/anime/poster_images/12/large.jpg",
    banner: "https://media.kitsu.app/anime/cover_images/12/large.jpg",
    synopsis: "Monkey D. Luffy et son équipage de pirates sillonnent les mers à la recherche du trésor légendaire, le One Piece.",
    score: "85.2%",
    genres: ["Action", "Aventure", "Comédie"],
    total_episodes_official: 1100,
    year: 1999,
  },
  "attack on titan": {
    title: "Attack on Titan (L'Attaque des Titans)",
    cover: "https://media.kitsu.app/anime/poster_images/7442/large.jpg",
    banner: "https://media.kitsu.app/anime/cover_images/7442/large.jpg",
    synopsis: "Dans un monde assiégé par de monstrueux Titans, Eren Jaeger s'enrôle dans le Bataillon d'exploration pour reconquérir la liberté.",
    score: "87.0%",
    genres: ["Action", "Drame", "Mystère"],
    total_episodes_official: 25,
    year: 2013,
  },
  "solo leveling": {
    title: "Solo Leveling",
    cover: "https://media.kitsu.app/anime/poster_images/46123/large.jpg",
    banner: "https://media.kitsu.app/anime/cover_images/46123/large.jpg",
    synopsis: "Sung Jinwoo, chasseur de rang E connu comme le plus faible de toute l'humanité, reçoit une quête secrète qui change son destin.",
    score: "86.4%",
    genres: ["Action", "Fantasy"],
    total_episodes_official: 12,
    year: 2024,
  },
  "demon slayer": {
    title: "Demon Slayer: Kimetsu no Yaiba",
    cover: "https://media.kitsu.app/anime/poster_images/41370/large.jpg",
    banner: "https://media.kitsu.app/anime/cover_images/41370/large.jpg",
    synopsis: "Tanjiro Kamado entreprend un voyage périlleux pour trouver un remède à la malédiction de sa sœur Nezuko, devenue démone.",
    score: "86.8%",
    genres: ["Action", "Démons", "Historique"],
    total_episodes_official: 26,
    year: 2019,
  },
};

// --- Anime Metadata Fetcher (Fast catalog + Kitsu API + AniList Fallback + Local SVG Fallback) ---
async function fetchAnimeMetadata(query: string): Promise<AnimeMetadata> {
  const cleanQ = query.trim();
  const lowerQ = cleanQ.toLowerCase();

  // 1. Instant match in popular catalog
  for (const [key, preset] of Object.entries(POPULAR_ANIME_CATALOG)) {
    if (lowerQ === key || lowerQ.includes(key) || key.includes(lowerQ)) {
      return {
        title: preset.title || cleanQ,
        cover: preset.cover || null,
        banner: preset.banner || null,
        synopsis: preset.synopsis || "Résumé non disponible.",
        score: preset.score || "85%",
        genres: preset.genres || ["Anime"],
        total_episodes_official: preset.total_episodes_official || 24,
        year: preset.year || 2024,
      };
    }
  }

  // 2. Try Kitsu API (Reliable, fast, returns real posters and ratings)
  try {
    const kitsuUrl = `https://kitsu.io/api/edge/anime?filter[text]=${encodeURIComponent(cleanQ)}&page[limit]=1`;
    const response = await fetch(kitsuUrl, {
      headers: { Accept: "application/vnd.api+json", "Content-Type": "application/vnd.api+json" },
      signal: AbortSignal.timeout(8000),
    });

    if (response.ok) {
      const json: any = await response.json();
      const item = json?.data?.[0]?.attributes;
      if (item) {
        return {
          title: item.canonicalTitle || cleanQ,
          cover: item.posterImage?.large || item.posterImage?.medium || item.posterImage?.original || null,
          banner: item.coverImage?.large || item.coverImage?.original || null,
          synopsis: item.synopsis || "Aucun résumé disponible.",
          score: item.averageRating ? `${parseFloat(item.averageRating).toFixed(1)}%` : "N/A",
          genres: ["Anime", "Shonen"],
          total_episodes_official: item.episodeCount ?? 24,
          year: item.startDate ? parseInt(item.startDate.split("-")[0], 10) : null,
        };
      }
    }
  } catch (err) {
    console.warn("Kitsu fetch failed or timed out:", err);
  }

  // 3. Try AniList as fallback
  try {
    const graphqlQuery = `
      query ($search: String) {
        Media (search: $search, type: ANIME) {
          title { romaji english }
          coverImage { extraLarge large }
          bannerImage
          description(asHtml: false)
          averageScore genres episodes status seasonYear
        }
      }
    `;
    const response = await fetch("https://graphql.anilist.co", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: graphqlQuery, variables: { search: cleanQ } }),
      signal: AbortSignal.timeout(3000),
    });

    if (response.ok) {
      const json: any = await response.json();
      const media = json?.data?.Media;
      if (media) {
        return {
          title: media.title?.english || media.title?.romaji || cleanQ,
          cover: media.coverImage?.extraLarge || media.coverImage?.large || null,
          banner: media.bannerImage || null,
          synopsis: media.description || "Aucun résumé disponible.",
          score: `${media.averageScore ?? "N/A"}%`,
          genres: media.genres || [],
          total_episodes_official: media.episodes ?? 24,
          year: media.seasonYear ?? null,
        };
      }
    }
  } catch (err) {
    console.warn("AniList fallback also failed:", err);
  }

  // 4. Guaranteed clean SVG fallback cover (never broken, loads instantly)
  const svgCover = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450"><rect width="300" height="450" fill="%23161b22"/><rect x="15" y="15" width="270" height="420" rx="8" fill="%230d1117" stroke="%2330363d" stroke-width="2"/><text x="150" y="210" fill="%2358a6ff" font-size="20" font-family="sans-serif" font-weight="bold" text-anchor="middle">NLSbox Anime</text><text x="150" y="245" fill="%23c9d1d9" font-size="14" font-family="sans-serif" text-anchor="middle">${encodeURIComponent(cleanQ.slice(0, 20))}</text></svg>`;

  return {
    title: cleanQ.charAt(0).toUpperCase() + cleanQ.slice(1),
    cover: svgCover,
    banner: null,
    synopsis: "Catalogue d'épisodes issus des canaux Telegram.",
    score: "85%",
    genres: ["Anime"],
    total_episodes_official: 24,
    year: 2024,
  };
}

// Real MP4 sample buffer generated with ffmpeg for testing Range requests and video streaming
function getSampleVideoBuffer(): Buffer {
  try {
    const samplePath = path.join(process.cwd(), "sample.mp4");
    if (fs.existsSync(samplePath)) {
      return fs.readFileSync(samplePath);
    }
  } catch (err) {
    console.warn("Could not read sample.mp4 from disk, falling back to generated buffer", err);
  }
  return Buffer.alloc(512 * 1024);
}

const sampleVideoData = getSampleVideoBuffer();

// --- ROUTES ---

// 1. Home endpoint - Always renders HTML interactive dashboard unless format=json is explicitly requested
app.get("/", (req: Request, res: Response) => {
  const accept = req.headers.accept || "";
  const wantsJson = req.query.format === "json" || (accept === "application/json" && !accept.includes("text/html"));
  if (wantsJson) {
    return res.json({ status: "En ligne - Sanitizer & Tri V2 OK", app: "NLSbox Backend Pro" });
  }

  res.setHeader("Content-Type", "text/html; charset=utf-8");
  return res.send(`
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Moteur de streaming et recherche d'animes connecté directement à Telegram MTProto avec support des requêtes HTTP Range 206 et dépollution des métadonnées">
  <title>NLSbox Pro Engine</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-hover: #79c0ff;
      --badge-bg: #238636;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 24px;
    }
    .container { max-width: 960px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    h1 { font-size: 24px; font-weight: 600; color: #fff; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 20px;
      background: var(--badge-bg);
      color: #fff;
      font-size: 12px;
      font-weight: 600;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .search-box {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    input {
      flex: 1;
      min-width: 200px;
      padding: 10px 14px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: #fff;
      font-size: 14px;
    }
    input:focus { outline: none; border-color: var(--accent); }
    button {
      padding: 10px 20px;
      background: #238636;
      border: none;
      border-radius: 6px;
      color: #fff;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #2ea043; }
    .api-preview {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      font-family: monospace;
      font-size: 13px;
      overflow-x: auto;
      color: #7ee787;
      margin-top: 12px;
    }
    .anime-card {
      display: flex;
      gap: 20px;
      margin-top: 20px;
    }
    .anime-card img {
      width: 140px;
      height: 200px;
      object-fit: cover;
      border-radius: 6px;
    }
    .anime-details h3 { color: #fff; margin-bottom: 8px; }
    .anime-details p { font-size: 14px; color: var(--text-muted); margin-bottom: 8px; }
    .episodes-list {
      margin-top: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .episode-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 14px;
      transition: border-color 0.2s;
    }
    .episode-item:hover {
      border-color: #388bfd;
    }
    .episode-header {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .tag-ep {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      background: #1f6feb22;
      color: #58a6ff;
      border: 1px solid #1f6feb55;
      font-weight: 600;
      font-size: 12px;
    }
    .tag-quality {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      background: #23863622;
      color: #3fb950;
      border: 1px solid #23863655;
      font-size: 11px;
      font-weight: 600;
    }
    .tag-lang {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      background: #a371f722;
      color: #d2a8ff;
      border: 1px solid #a371f755;
      font-size: 11px;
      font-weight: 600;
    }
    .raw-info {
      font-size: 11px;
      color: #8b949e;
      margin-top: 4px;
      font-family: monospace;
      word-break: break-all;
    }
    .episode-link {
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      margin-left: 12px;
    }
    .episode-link:hover { text-decoration: underline; }
    .quick-chips {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .chip {
      padding: 4px 10px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 16px;
      color: #c9d1d9;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.2s;
    }
    .chip:hover {
      background: #30363d;
      color: #fff;
    }
    .feature-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      background: #161b22;
      border: 1px solid #30363d;
      font-size: 12px;
      color: #8b949e;
    }
    .feature-badge strong {
      color: #58a6ff;
    }
    video {
      width: 100%;
      max-height: 360px;
      background: #000;
      border-radius: 6px;
      margin-top: 12px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>NLSbox Pro Backend Engine</h1>
        <p style="color: var(--text-muted); font-size: 13px;">Streaming & Téléchargement Universel Telegram : Musique, Vidéos, Films, Séries, Scans, Documents</p>
      </div>
      <div>
        <span class="badge">MTProto Direct • Range 206 Actif</span>
      </div>
    </header>

    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;">
      <div class="feature-badge">
        <span>⚡</span> <strong>Streaming Turbo 206</strong> (Scrubbing instantané)
      </div>
      <div class="feature-badge">
        <span>🎵</span> <strong>Lecteur Audio & Vidéo</strong> (MP3, MP4, FLAC, MKV)
      </div>
      <div class="feature-badge">
        <span>📥</span> <strong>Téléchargement Direct</strong> (Fichiers d'origine préservés)
      </div>
      <div class="feature-badge">
        <span>📁</span> <strong>Tous Canaux Telegram</strong> (Multimédia sans limite)
      </div>
    </div>

    <div id="tgAccountBox" style="margin-bottom: 20px; padding: 14px 18px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; font-size: 13px;">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <span id="tgAccountDot" style="height: 10px; width: 10px; border-radius: 50%; background: #e3b341; display: inline-block; box-shadow: 0 0 8px #e3b341;"></span>
          <strong id="tgAccountText" style="color: #fff;">Telegram MTProto : Connexion en cours...</strong>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
          <span id="tgAccountDetails" style="color: var(--text-muted); font-family: monospace;">Chargement de la session...</span>
          <button id="tgManageBtn" onclick="toggleSessionModal(true)" style="padding: 4px 10px; font-size: 12px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">⚙️ Gérer Session</button>
        </div>
      </div>

      <!-- Alerte Conflit MTProto 406 AUTH_KEY_DUPLICATED -->
      <div id="tgConflictAlert" style="display: none; margin-top: 14px; padding: 12px 14px; background: #2b1111; border: 1px solid #f85149; border-radius: 6px; color: #f0f6fc; font-size: 13px;">
        <div style="display: flex; align-items: flex-start; gap: 10px;">
          <span style="font-size: 20px; line-height: 1;">⚠️</span>
          <div style="flex: 1;">
            <strong style="color: #ff7b72; font-size: 14px;">Conflit Telegram 406 : AUTH_KEY_DUPLICATED</strong>
            <p style="margin: 6px 0; color: #c9d1d9; line-height: 1.5;">
              Telegram a révoqué cette session (<code>SESSION_STRING</code>) car elle a été connectée simultanément depuis deux instances (ex: votre déploiement <strong>Render</strong> et cet environnement de test, ou deux processus). Telegram invalide immédiatement toute clé utilisée en double.
            </p>
            <div style="background: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 6px; margin: 8px 0; font-size: 12px;">
              <div style="font-weight: 600; color: #58a6ff; margin-bottom: 4px;">Comment résoudre en 1 minute :</div>
              <ol style="padding-left: 20px; color: #8b949e; line-height: 1.6;">
                <li>Exécutez <code style="background: #0d1117; color: #7ee787; padding: 2px 6px; border-radius: 3px;">npm run session</code> dans le terminal pour générer une nouvelle <strong>SESSION_STRING</strong> Telegram.</li>
                <li>Cliquez sur <strong>Remplacer la SESSION_STRING</strong> ci-dessous pour coller la nouvelle clé sans redémarrer le serveur.</li>
                <li>Le catalogue de démonstration et les filtres de recherche restent actifs en attendant.</li>
              </ol>
            </div>
            <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
              <button onclick="toggleSessionModal(true)" style="padding: 6px 14px; background: #238636; color: #fff; border: none; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer;">🔑 Remplacer la SESSION_STRING</button>
              <button onclick="retryTelegramConnection()" style="padding: 6px 14px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; font-size: 12px; cursor: pointer;">🔄 Réessayer la connexion</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal Mise à jour SESSION_STRING -->
    <div id="sessionModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 9999; align-items: center; justify-content: center; padding: 20px;">
      <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; width: 100%; max-width: 520px; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
          <h3 style="color: #fff; font-size: 16px;">🔑 Gestionnaire de Session Telegram MTProto</h3>
          <button onclick="toggleSessionModal(false)" style="background: none; border: none; color: var(--text-muted); font-size: 20px; cursor: pointer;">&times;</button>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
          Collez votre nouvelle <strong>SESSION_STRING</strong> Telegram (générée avec <code style="color: #58a6ff;">npm run session</code>). Le serveur testera et reconnectera instantanément le moteur MTProto.
        </p>
        <textarea id="newSessionInput" placeholder="Collez la SESSION_STRING ici (ex: 1BVtsOHQ...)" rows="4" style="width: 100%; background: #0d1117; border: 1px solid var(--border); border-radius: 6px; color: #fff; font-family: monospace; font-size: 12px; padding: 10px; resize: vertical; margin-bottom: 14px;"></textarea>
        <div id="sessionStatusMsg" style="font-size: 13px; margin-bottom: 12px; display: none;"></div>
        <div style="display: flex; justify-content: flex-end; gap: 10px;">
          <button onclick="toggleSessionModal(false)" style="padding: 8px 16px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; font-size: 13px; cursor: pointer;">Fermer</button>
          <button id="saveSessionBtn" onclick="saveNewSession()" style="padding: 8px 16px; background: #238636; color: #fff; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;">Tester & Connecter</button>
        </div>
      </div>
    </div>

    <!-- Modal Reconnaissance d'Affiche Telegram (Gemini Vision) -->
    <div id="visionModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 9999; align-items: center; justify-content: center; padding: 20px;">
      <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; width: 100%; max-width: 580px; padding: 22px; max-height: 90vh; overflow-y: auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
          <h3 style="color: #fff; font-size: 16px; display: flex; align-items: center; gap: 8px; margin: 0;">
            <span>🖼️</span> Reconnaissance d'Affiche (Gemini Multimodal Vision)
          </h3>
          <button onclick="toggleVisionModal(false)" style="background: none; border: none; color: var(--text-muted); font-size: 20px; cursor: pointer;">&times;</button>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px; line-height: 1.5;">
          Quand un épisode Telegram est publié avec un nom générique sans titre (ex: <code style="color: #58a6ff;">01.mp4</code> ou <code style="color: #58a6ff;">ep02.mkv</code>), Gemini Vision inspecte l'affiche Telegram pour repérer la série, la saison et le synopsis.
        </p>

        <div style="margin-bottom: 14px;">
          <label style="font-size: 12px; color: #8b949e; display: block; margin-bottom: 6px;">1. Importer une affiche / jaquette :</label>
          <input type="file" id="posterFileInput" accept="image/*" onchange="onPosterFileSelected(event)" style="width: 100%; font-size: 12px; color: #c9d1d9; background: #0d1117; padding: 8px; border: 1px solid #30363d; border-radius: 6px;">
        </div>

        <div style="margin-bottom: 14px;">
          <label style="font-size: 12px; color: #8b949e; display: block; margin-bottom: 6px;">Ou tester avec des affiches types :</label>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <button type="button" onclick="loadSamplePoster('death_note')" style="padding: 5px 10px; font-size: 12px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">Death Note</button>
            <button type="button" onclick="loadSamplePoster('jujutsu')" style="padding: 5px 10px; font-size: 12px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">Jujutsu Kaisen</button>
            <button type="button" onclick="loadSamplePoster('solo_leveling')" style="padding: 5px 10px; font-size: 12px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">Solo Leveling</button>
          </div>
        </div>

        <div style="display: flex; gap: 10px; margin-bottom: 14px;">
          <input type="text" id="visionFileNameHint" placeholder="Nom du fichier (ex: 01.mp4)" style="flex: 1; padding: 8px; font-size: 12px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #fff;">
          <input type="text" id="visionCaptionHint" placeholder="Légende Telegram (optionnel)" style="flex: 1; padding: 8px; font-size: 12px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; color: #fff;">
        </div>

        <!-- Aperçu de l'affiche -->
        <div id="posterPreviewContainer" style="display: none; text-align: center; margin-bottom: 14px; background: #0d1117; padding: 10px; border-radius: 6px; border: 1px dashed #30363d;">
          <img id="posterPreviewImg" src="" alt="Aperçu affiche" style="max-height: 180px; max-width: 100%; border-radius: 4px; object-fit: contain;">
        </div>

        <!-- Résultat de l'analyse Vision -->
        <div id="visionResultBox" style="display: none; padding: 14px; background: #12251a; border: 1px solid #238636; border-radius: 6px; margin-bottom: 14px; font-size: 13px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <strong style="color: #7ee787; font-size: 15px;" id="visionAnimeTitle">Anime Détecté</strong>
            <span id="visionConfidence" style="font-size: 11px; background: #238636; color: #fff; padding: 2px 6px; border-radius: 4px;">100% confiance</span>
          </div>
          <div style="color: #c9d1d9; margin-bottom: 6px; font-size: 13px;" id="visionAnimeDetails">Saison & Épisode</div>
          <div style="color: #8b949e; font-size: 12px; margin-bottom: 8px; line-height: 1.5;" id="visionAnimeSynopsis"></div>
          <div style="display: flex; gap: 6px; flex-wrap: wrap;" id="visionAnimeGenres"></div>
        </div>

        <div id="visionErrorBox" style="display: none; padding: 10px; background: #3d1b22; border: 1px solid #da3633; border-radius: 6px; color: #ff7b72; font-size: 12px; margin-bottom: 14px;"></div>

        <div style="display: flex; justify-content: flex-end; gap: 10px;">
          <button onclick="toggleVisionModal(false)" style="padding: 8px 16px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; font-size: 13px; cursor: pointer;">Fermer</button>
          <button id="runVisionBtn" onclick="runPosterIdentification()" style="padding: 8px 16px; background: #1f6feb; color: #fff; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;">🔍 Identifier l'Anime</button>
        </div>
      </div>
    </div>

    <!-- Modal Recherche Intelligente IA -->
    <div id="smartSearchModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 9999; align-items: center; justify-content: center; padding: 20px;">
      <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; width: 100%; max-width: 600px; padding: 22px; max-height: 90vh; overflow-y: auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
          <h3 style="color: #fff; font-size: 16px; display: flex; align-items: center; gap: 8px; margin: 0;">
            <span>🧠</span> Recherche Sémantique & Recommandations IA (Gemini)
          </h3>
          <button onclick="toggleSmartSearchModal(false)" style="background: none; border: none; color: var(--text-muted); font-size: 20px; cursor: pointer;">&times;</button>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 14px;">
          Exprimez votre recherche librement en français : description d'une scène, anime d'un genre précis, saison spécifique ou version linguistique (VF/VOSTFR).
        </p>

        <div style="display: flex; gap: 8px; margin-bottom: 14px;">
          <input type="text" id="smartSearchInput" placeholder="Ex: anime d'exorcisme avec des démons, saison 2 en VF..." style="flex: 1; padding: 10px 14px; background: #0d1117; border: 1px solid var(--border); color: #fff; border-radius: 6px; font-size: 13px;" onkeydown="if(event.key==='Enter') runSmartSearch()">
          <button id="runSmartSearchBtn" onclick="runSmartSearch()" style="padding: 10px 18px; background: #238636; color: #fff; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;">Analyser</button>
        </div>

        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px;">
          <span style="font-size: 11px; color: #8b949e; align-self: center;">Suggestions rapides :</span>
          <button type="button" onclick="setSmartSearchQuery('anime sombre de manipulation psychologique')" style="padding: 3px 8px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">Thriller psychologique</button>
          <button type="button" onclick="setSmartSearchQuery('Solo Leveling épisode de montée en niveau')" style="padding: 3px 8px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">Solo Leveling</button>
          <button type="button" onclick="setSmartSearchQuery('combats shonen avec malédictions saison 2')" style="padding: 3px 8px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 4px; cursor: pointer;">Exorcisme Shonen</button>
        </div>

        <!-- Résultat IA Smart Search -->
        <div id="smartSearchResultArea" style="display: none; padding: 14px; background: #0f1d14; border: 1px solid #238636; border-radius: 6px; margin-bottom: 14px; font-size: 13px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="color: #7ee787; font-size: 14px;">🎯 Intention Comprise par l'IA</strong>
            <span id="smartSearchTargetBadge" style="background: #238636; color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: bold;"></span>
          </div>
          <div id="smartSearchIntentDesc" style="color: #c9d1d9; margin-bottom: 10px; font-size: 13px; line-height: 1.4;"></div>
          
          <div style="margin-bottom: 10px;">
            <span style="font-size: 12px; color: #8b949e;">Mots-clés optimaux générés :</span>
            <div id="smartSearchKeywords" style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px;"></div>
          </div>

          <div id="smartSearchRecsContainer" style="margin-top: 12px; border-top: 1px solid #1f3d27; padding-top: 10px;">
            <div style="font-size: 12px; font-weight: bold; color: #7ee787; margin-bottom: 6px;">✨ Animes Recommandés :</div>
            <div id="smartSearchRecsList" style="display: flex; flex-direction: column; gap: 6px;"></div>
          </div>

          <div style="margin-top: 14px; text-align: right;">
            <button id="applySmartSearchBtn" onclick="applySmartSearchAndClose()" style="padding: 8px 16px; background: #1f6feb; border: 1px solid #388bfd; color: #fff; border-radius: 6px; font-size: 12px; font-weight: bold; cursor: pointer;">🔍 Lancer cette recherche dans Telegram</button>
          </div>
        </div>

        <div id="smartSearchErrorBox" style="display: none; padding: 10px; background: #3d1b22; border: 1px solid #da3633; border-radius: 6px; color: #ff7b72; font-size: 12px; margin-bottom: 14px;"></div>

        <div style="display: flex; justify-content: flex-end;">
          <button onclick="toggleSmartSearchModal(false)" style="padding: 8px 16px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; font-size: 13px; cursor: pointer;">Fermer</button>
        </div>
      </div>
    </div>

    <!-- Modal Assistant Conversationnel NLSbox IA -->
    <div id="chatModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 9999; align-items: center; justify-content: center; padding: 20px;">
      <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; width: 100%; max-width: 680px; height: 85vh; display: flex; flex-direction: column; overflow: hidden;">
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid #30363d; background: #0d1117;">
          <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">💬</span>
            <div>
              <h3 style="color: #fff; font-size: 15px; margin: 0;">Assistant IA NLSbox (Gemini 3.8 Flash)</h3>
              <span style="font-size: 11px; color: #8b949e;">Conseils, ordres de visionnage, arcs canon/filler & recommandations</span>
            </div>
          </div>
          <button onclick="toggleChatModal(false)" style="background: none; border: none; color: var(--text-muted); font-size: 22px; cursor: pointer;">&times;</button>
        </div>

        <!-- Chat messages container -->
        <div id="chatMessages" style="flex: 1; overflow-y: auto; padding: 18px 20px; display: flex; flex-direction: column; gap: 14px;">
          <div style="background: #1e2632; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; max-width: 85%; align-self: flex-start; font-size: 13px; line-height: 1.5; color: #c9d1d9;">
            👋 <strong>Bonjour ! Je suis l'assistant IA de NLSbox.</strong><br>
            Je peux vous aider à choisir quoi regarder, expliquer l'ordre chronologique d'un anime, vous dire quels épisodes sont des fillers ou explorer vos chaînes Telegram. Que souhaitez-vous savoir ?
          </div>
        </div>

        <!-- Quick prompts -->
        <div style="padding: 8px 16px; background: #0d1117; border-top: 1px solid #21262d; display: flex; gap: 6px; overflow-x: auto; white-space: nowrap;">
          <button type="button" onclick="sendQuickPrompt(&quot;Ordre de visionnage pour Bleach&quot;)" style="padding: 4px 10px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #58a6ff; border-radius: 12px; cursor: pointer;">Ordre Bleach</button>
          <button type="button" onclick="sendQuickPrompt('Recommande-moi un anime dans le même esprit que Solo Leveling')" style="padding: 4px 10px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #58a6ff; border-radius: 12px; cursor: pointer;">Comme Solo Leveling</button>
          <button type="button" onclick="sendQuickPrompt('Y a-t-il des épisodes fillers à sauter dans Death Note ?')" style="padding: 4px 10px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #58a6ff; border-radius: 12px; cursor: pointer;">Fillers Death Note ?</button>
          <button type="button" onclick="sendQuickPrompt('Explique-moi les saisons et le film de Jujutsu Kaisen')" style="padding: 4px 10px; font-size: 11px; background: #21262d; border: 1px solid #30363d; color: #58a6ff; border-radius: 12px; cursor: pointer;">Saisons Jujutsu Kaisen</button>
        </div>

        <!-- Input area -->
        <div style="padding: 14px 16px; background: #0d1117; border-top: 1px solid #30363d; display: flex; gap: 10px;">
          <input type="text" id="chatInput" placeholder="Posez une question sur un anime, un ordre de visionnage..." style="flex: 1; padding: 10px 14px; background: #161b22; border: 1px solid var(--border); color: #fff; border-radius: 6px; font-size: 13px;" onkeydown="if(event.key==='Enter') sendChatMessage()">
          <button id="sendChatBtn" onclick="sendChatMessage()" style="padding: 10px 18px; background: #8957e5; color: #fff; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;">Envoyer</button>
        </div>
      </div>
    </div>

    <!-- Modal Fiche & Métadonnées IA -->
    <div id="enrichModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 9999; align-items: center; justify-content: center; padding: 20px;">
      <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; width: 100%; max-width: 580px; padding: 22px; max-height: 85vh; overflow-y: auto;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
          <div>
            <h3 id="enrichCanonicalTitle" style="color: #fff; font-size: 18px; margin: 0;">Fiche Anime</h3>
            <div id="enrichOriginalTitle" style="color: #8b949e; font-size: 12px; margin-top: 2px;"></div>
          </div>
          <button onclick="toggleEnrichModal(false)" style="background: none; border: none; color: var(--text-muted); font-size: 20px; cursor: pointer;">&times;</button>
        </div>

        <div id="enrichLoading" style="padding: 24px; text-align: center; color: #58a6ff; font-size: 14px;">
          Génération de la fiche encyclopédique par Gemini IA...
        </div>

        <div id="enrichContent" style="display: none;">
          <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;">
            <span id="enrichYear" style="background: #21262d; border: 1px solid #30363d; color: #7ee787; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: bold;"></span>
            <span id="enrichStudio" style="background: #21262d; border: 1px solid #30363d; color: #58a6ff; font-size: 11px; padding: 3px 8px; border-radius: 4px;"></span>
            <span id="enrichRating" style="background: #21262d; border: 1px solid #30363d; color: #e3b341; font-size: 11px; padding: 3px 8px; border-radius: 4px;"></span>
            <span id="enrichEpisodesCount" style="background: #21262d; border: 1px solid #30363d; color: #a371f7; font-size: 11px; padding: 3px 8px; border-radius: 4px;"></span>
          </div>

          <div style="margin-bottom: 14px;">
            <label style="font-size: 12px; color: #8b949e; font-weight: bold; text-transform: uppercase;">Genres</label>
            <div id="enrichGenres" style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px;"></div>
          </div>

          <div style="margin-bottom: 14px;">
            <label style="font-size: 12px; color: #8b949e; font-weight: bold; text-transform: uppercase;">Synopsis Officiel</label>
            <p id="enrichSynopsis" style="font-size: 13px; color: #c9d1d9; line-height: 1.6; margin-top: 4px; background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #21262d;"></p>
          </div>

          <div style="margin-bottom: 14px;">
            <label style="font-size: 12px; color: #8b949e; font-weight: bold; text-transform: uppercase;">Conseil Ordre de Visionnage</label>
            <p id="enrichWatchOrder" style="font-size: 13px; color: #7ee787; line-height: 1.5; margin-top: 4px; background: #0d1b14; padding: 10px 12px; border-radius: 6px; border: 1px solid #238636;"></p>
          </div>
        </div>

        <div style="display: flex; justify-content: flex-end; margin-top: 14px;">
          <button onclick="toggleEnrichModal(false)" style="padding: 8px 16px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; font-size: 13px; cursor: pointer;">Fermer</button>
        </div>
      </div>
    </div>

    <!-- Module IA Gemini Status -->
    <div id="aiStatusBox" style="margin-bottom: 20px; padding: 14px 18px; background: #0f1d14; border: 1px solid #238636; border-radius: 8px; font-size: 13px;">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <span style="font-size: 24px;">🤖</span>
          <div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <strong style="color: #7ee787; font-size: 14px;">Module IA Gemini 3.8 Flash</strong>
              <span id="aiBadge" style="background: #238636; color: #fff; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold;">ACTIF</span>
            </div>
            <div id="aiStatusDesc" style="color: #8b949e; font-size: 12px; margin-top: 2px;">
              Déduplication S01E01, recherche sémantique en langage naturel, vision d'affiche & assistant chat.
            </div>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <button type="button" onclick="toggleSmartSearchModal(true)" style="padding: 6px 12px; font-size: 12px; background: #238636; border: 1px solid #2ea043; color: #fff; border-radius: 6px; cursor: pointer; font-weight: 500; display: inline-flex; align-items: center; gap: 4px;">🧠 Recherche IA</button>
          <button type="button" onclick="toggleChatModal(true)" style="padding: 6px 12px; font-size: 12px; background: #8957e5; border: 1px solid #a371f7; color: #fff; border-radius: 6px; cursor: pointer; font-weight: 500; display: inline-flex; align-items: center; gap: 4px;">💬 Assistant NLSbox</button>
          <button type="button" onclick="toggleVisionModal(true)" style="padding: 6px 12px; font-size: 12px; background: #1f6feb; border: 1px solid #388bfd; color: #fff; border-radius: 6px; cursor: pointer; font-weight: 500; display: inline-flex; align-items: center; gap: 4px;">🖼️ Vision d'Affiche</button>
        </div>
      </div>
    </div>

    <div id="errorBox" style="display: none; padding: 12px 16px; background: #3d1b22; border: 1px solid #da3633; border-radius: 6px; color: #ff7b72; font-size: 14px; margin-bottom: 20px;"></div>

    <div class="card">
      <h2 style="font-size: 16px; margin-bottom: 12px; color: #fff;">Exploration & Recherche Multimédia Telegram</h2>
      
      <div class="search-box">
        <input type="text" id="queryInput" value="" placeholder="Rechercher un fichier, musique, film, série, scan... (ou vide pour tout voir)">
        
        <select id="channelSelect" onchange="onChannelSelected(this.value)" style="padding: 10px 14px; background: #0d1117; border: 1px solid var(--border); color: #c9d1d9; border-radius: 6px; font-size: 14px; min-width: 220px; max-width: 320px;">
          <option value="-1004272203145" selected>📁 NLSbox (-1004272203145)</option>
          <option value="-1003914934147">🎵 NLSmusic1 (-1003914934147)</option>
          <option value="-1003222560776">🎵 Music World (-1003222560776)</option>
          <option value="-1001558851926">🎬 Anime zone VF (-1001558851926)</option>
          <option value="-1002297137971">🎬 Ciné+ VF (-1002297137971)</option>
          <option value="-1001120630831">📄 Scan Zone (-1001120630831)</option>
          <option value="custom">✏️ Saisir un autre ID ou @canal...</option>
        </select>

        <input type="text" id="channelInput" value="-1004272203145" placeholder="ID (ex: -1004272203145) ou @canal..." style="max-width: 220px; display: none;">
        <button id="searchBtn" onclick="runSearch()">Explorer</button>
      </div>

      <!-- Option de déduplication et ordonnancement IA -->
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-top: 14px; padding: 10px 14px; background: #161b22; border: 1px solid #30363d; border-radius: 6px;">
        <label style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: #c9d1d9; cursor: pointer; user-select: none;">
          <input type="checkbox" id="dedupToggle" checked onchange="runSearch()" style="accent-color: #238636; width: 16px; height: 16px; cursor: pointer;">
          <span><strong>Déduplication & Ordonnancement IA</strong> (Fusionner 1080p / 720p et normaliser les titres en S01E01...)</span>
        </label>
        <span id="dedupStatsBadge" style="display: none; font-size: 12px; background: #1b4722; color: #7ee787; border: 1px solid #238636; padding: 3px 10px; border-radius: 12px; font-weight: 500;"></span>
      </div>

      <!-- Filtres par catégorie -->
      <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px;">
        <button type="button" class="filter-tab active" id="tab-all" onclick="setFilterType('all')">🌐 Tous les fichiers</button>
        <button type="button" class="filter-tab" id="tab-video" onclick="setFilterType('video')">🎬 Vidéos & Films</button>
        <button type="button" class="filter-tab" id="tab-audio" onclick="setFilterType('audio')">🎵 Musique & Audio</button>
        <button type="button" class="filter-tab" id="tab-document" onclick="setFilterType('document')">📄 Scans & Documents</button>
        <button type="button" class="filter-tab" id="tab-archive" onclick="setFilterType('archive')">📦 Archives & Fichiers</button>
      </div>

      <div class="quick-chips" style="margin-top: 14px;">
        <span style="font-size: 12px; color: var(--text-muted); align-self: center;">Raccourcis rapides :</span>
        <div class="chip" onclick="quickSearch('', '-1004272203145')">📁 Tout NLSbox</div>
        <div class="chip" onclick="quickSearch('', '-1003914934147')">🎵 Tout NLSmusic1</div>
        <div class="chip" onclick="quickSearch('Death Note', '-1004272203145')">Death Note</div>
        <div class="chip" onclick="quickSearch('Ninho', '-1003222560776')">Ninho (Music)</div>
        <div class="chip" onclick="quickSearch('Bleach', '-1003402221387')">Bleach VF</div>
      </div>

      <!-- Zone de lecture active -->
      <div id="playerSection" class="card" style="margin-top: 20px; display: none; background: #11161d; border-color: #388bfd;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; animation: pulse 1.5s infinite;"></span>
            <strong id="nowPlayingTitle" style="color: #fff; font-size: 15px;">En cours de lecture</strong>
          </div>
          <a id="nowPlayingDownloadBtn" href="javascript:void(0)" class="episode-link" style="font-size: 12px; padding: 5px 12px; display: none;">📥 Télécharger ce fichier</a>
        </div>

        <!-- Lecteur Vidéo -->
        <video id="videoPlayer" controls playsinline preload="auto" style="width: 100%; border-radius: 6px; background: #000; max-height: 480px; display: none;">
          Votre navigateur ne supporte pas la balise vidéo.
        </video>

        <!-- Lecteur Audio -->
        <div id="audioContainer" style="display: none; padding: 16px; background: #161b22; border-radius: 8px; border: 1px solid #30363d;">
          <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px;">
            <span style="font-size: 32px;">🎵</span>
            <div>
              <div id="audioTrackName" style="color: #fff; font-weight: bold; font-size: 15px;">Piste audio</div>
              <div id="audioTrackArtist" style="color: #58a6ff; font-size: 13px;">Artiste Telegram</div>
            </div>
          </div>
          <audio id="audioPlayer" controls style="width: 100%; border-radius: 4px; outline: none;"></audio>
        </div>
      </div>

      <div id="resultsArea" style="display: none; margin-top: 20px;">
        <!-- MovieBox Series Hero Banner -->
        <div id="movieboxHero" class="moviebox-hero" style="display: none;"></div>

        <!-- MovieBox Season Selector Tabs -->
        <div id="movieboxSeasonsBar" class="moviebox-seasons" style="display: none;"></div>

        <!-- MovieBox Quick Episode Grid -->
        <div id="movieboxQuickGrid" class="moviebox-quick-grid" style="display: none;"></div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
          <h4 style="color: #fff; margin: 0; font-size: 15px;">
            Contenus ordonnés (<span id="epCount">0</span>)
          </h4>
          <div style="display: flex; gap: 8px; align-items: center;">
            <button id="sortToggleBtn" type="button" onclick="toggleSortOrder()" style="background: #21262d; border: 1px solid #30363d; color: #8b949e; font-size: 11px; padding: 4px 10px; border-radius: 4px; cursor: pointer;">↕️ Tri : Épisode 1 → Fin</button>
            <span id="categoryLabel" style="font-size: 12px; color: #58a6ff; background: #388bfd1a; padding: 2px 10px; border-radius: 4px; border: 1px solid #388bfd44;">Tous types</span>
          </div>
        </div>

        <div class="episodes-list" id="epList"></div>

        <!-- MovieBox Pagination Bar -->
        <div id="paginationBar" class="moviebox-pagination" style="display: none;"></div>
      </div>
    </div>

    <div class="card">
      <h2 style="font-size: 16px; margin-bottom: 8px; color: #fff;">Endpoints API Documentés</h2>
      <ul style="font-size: 14px; padding-left: 20px; color: var(--text-muted);">
        <li><code style="color: #58a6ff;">GET /channels</code> - Liste de tous les canaux Telegram de la session active</li>
        <li><code style="color: #58a6ff;">GET /search?q={query}&channel={channel_id}&type={all|video|audio|document}&page={page}</code> - Exploration universelle Telegram</li>
        <li><code style="color: #58a6ff;">GET /download/{channel_id}/{message_id}</code> - Flux streaming direct Range 206 pour lecteurs vidéo et audio</li>
        <li><code style="color: #58a6ff;">GET /download/{channel_id}/{message_id}?dl=1</code> - Téléchargement direct avec Content-Disposition: attachment</li>
      </ul>
      <div class="api-preview">
curl "http://localhost:3000/search?channel=-1004272203145"
curl -I "http://localhost:3000/download/-1004272203145/889" -H "Range: bytes=0-1048575"
      </div>
    </div>
  </div>

  <style>
    .filter-tab {
      padding: 6px 14px;
      font-size: 13px;
      background: #161b22;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .filter-tab:hover {
      border-color: #58a6ff;
      color: #fff;
    }
    .filter-tab.active {
      background: #1f6feb;
      border-color: #388bfd;
      color: #fff;
      font-weight: 500;
    }
    .tag-type {
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: bold;
      text-transform: uppercase;
      margin-right: 6px;
    }
    .tag-video { background: #238636; color: #fff; }
    .tag-audio { background: #8957e5; color: #fff; }
    .tag-doc { background: #da3633; color: #fff; }
    .tag-file { background: #6e7681; color: #fff; }
    .tag-variants {
      background: #238636;
      color: #fff;
      font-size: 10px;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: bold;
      margin-left: 6px;
    }
    .variant-btn {
      background: #21262d;
      border: 1px solid #30363d;
      color: #58a6ff;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.2s;
    }
    .variant-btn:hover {
      background: #30363d;
      border-color: #58a6ff;
      color: #fff;
    }
    /* MovieBox Layout & Components */
    .moviebox-hero {
      background: linear-gradient(180deg, #1b2230 0%, #161b22 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      display: flex;
      gap: 16px;
      align-items: flex-start;
    }
    .moviebox-poster {
      width: 105px;
      height: 150px;
      object-fit: cover;
      border-radius: 8px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.6);
      flex-shrink: 0;
      background: #0d1117;
      border: 1px solid #30363d;
    }
    .moviebox-info {
      flex: 1;
      min-width: 0;
    }
    .moviebox-title {
      font-size: 18px;
      font-weight: 700;
      color: #fff;
      margin: 0 0 6px 0;
    }
    .moviebox-meta {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
      font-size: 12px;
    }
    .moviebox-score {
      background: #d2992226;
      color: #e3b341;
      border: 1px solid #d2992266;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: bold;
    }
    .moviebox-year, .moviebox-eps-badge {
      background: #388bfd1a;
      color: #58a6ff;
      border: 1px solid #388bfd33;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .moviebox-synopsis {
      font-size: 13px;
      color: #8b949e;
      line-height: 1.5;
      margin-bottom: 10px;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .moviebox-seasons {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 6px;
      margin-bottom: 12px;
    }
    .moviebox-season-tab {
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      font-size: 12px;
      padding: 6px 14px;
      border-radius: 20px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .moviebox-season-tab:hover {
      border-color: #58a6ff;
      color: #fff;
    }
    .moviebox-season-tab.active {
      background: #1f6feb;
      border-color: #388bfd;
      color: #fff;
      font-weight: 600;
    }
    .moviebox-quick-grid {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding: 10px;
      background: #161b22;
      border-radius: 8px;
      border: 1px solid #21262d;
      max-height: 120px;
      overflow-y: auto;
    }
    .moviebox-ep-btn {
      min-width: 38px;
      height: 36px;
      padding: 0 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 6px;
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      font-size: 12px;
      font-weight: bold;
      cursor: pointer;
      transition: all 0.2s;
    }
    .moviebox-ep-btn:hover {
      background: #30363d;
      border-color: #58a6ff;
      color: #fff;
      transform: translateY(-1px);
    }
    .moviebox-ep-btn.active {
      background: #238636;
      border-color: #3fb950;
      color: #fff;
    }
    .moviebox-pagination {
      margin-top: 18px;
      padding: 12px 16px;
      background: #161b22;
      border-radius: 8px;
      border: 1px solid #30363d;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(63, 185, 80, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
    }
  </style>

  <script>
    let currentFilterType = 'all';
    let currentVisionBase64 = null;
    let currentVisionMime = 'image/jpeg';

    function setFilterType(type) {
      currentFilterType = type;
      document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('tab-' + type);
      if (activeBtn) activeBtn.classList.add('active');
      runSearch();
    }

    function showError(msg) {
      const errBox = document.getElementById('errorBox');
      if (errBox) {
        errBox.textContent = msg;
        errBox.style.display = 'block';
      }
    }
    function clearError() {
      const errBox = document.getElementById('errorBox');
      if (errBox) {
        errBox.style.display = 'none';
        errBox.textContent = '';
      }
    }

    // --- GESTION DU MODULE IA GEMINI ---
    let chatConversationHistory = [];
    let lastSmartSearchKeywords = [];

    function toggleSmartSearchModal(show) {
      const modal = document.getElementById('smartSearchModal');
      if (modal) modal.style.display = show ? 'flex' : 'none';
      if (show) {
        setTimeout(() => document.getElementById('smartSearchInput')?.focus(), 100);
      }
    }

    function setSmartSearchQuery(q) {
      const input = document.getElementById('smartSearchInput');
      if (input) {
        input.value = q;
        runSmartSearch();
      }
    }

    async function runSmartSearch() {
      const input = document.getElementById('smartSearchInput');
      const btn = document.getElementById('runSmartSearchBtn');
      const resArea = document.getElementById('smartSearchResultArea');
      const errBox = document.getElementById('smartSearchErrorBox');

      const query = (input?.value || '').trim();
      if (!query) return;

      if (errBox) errBox.style.display = 'none';
      if (resArea) resArea.style.display = 'none';
      if (btn) { btn.disabled = true; btn.textContent = 'Analyse Gemini...'; }

      try {
        const res = await fetch('/api/ai/smart-search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Erreur lors de l'analyse sémantique");

        const r = data.result || {};
        const targetBadge = document.getElementById('smartSearchTargetBadge');
        if (targetBadge) {
          targetBadge.textContent = (r.target_series || 'Contenu') + (r.season ? ' • S' + r.season : '') + (r.episode ? ' • E' + r.episode : '') + (r.language ? ' • ' + r.language : '');
        }

        const intentDesc = document.getElementById('smartSearchIntentDesc');
        if (intentDesc) {
          intentDesc.innerHTML = '<strong>Interprétation :</strong> ' + (r.interpreted_intent || 'Recherche de contenu');
        }

        const kwDiv = document.getElementById('smartSearchKeywords');
        lastSmartSearchKeywords = r.search_keywords || [r.target_series];
        if (kwDiv) {
          kwDiv.innerHTML = '';
          lastSmartSearchKeywords.forEach(k => {
            const sp = document.createElement('span');
            sp.style.cssText = 'background: #21262d; border: 1px solid #30363d; color: #58a6ff; font-size: 11px; padding: 2px 7px; border-radius: 4px;';
            sp.textContent = k;
            kwDiv.appendChild(sp);
          });
        }

        const recsList = document.getElementById('smartSearchRecsList');
        if (recsList) {
          recsList.innerHTML = '';
          (r.recommendations || []).forEach(rec => {
            const itemDiv = document.createElement('div');
            itemDiv.style.cssText = 'background: #161b22; border: 1px solid #30363d; padding: 8px 10px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; gap: 8px;';
            itemDiv.innerHTML = '<div style="flex: 1; min-width: 0;"><strong class="rec-t" style="color: #fff; font-size: 13px;"></strong><div class="rec-r" style="color: #8b949e; font-size: 11px; margin-top: 2px;"></div></div><button type="button" class="rec-btn" style="padding: 4px 8px; font-size: 11px; background: #21262d; border: 1px solid #388bfd; color: #58a6ff; border-radius: 4px; cursor: pointer; white-space: nowrap;">Chercher</button>';
            const tEl = itemDiv.querySelector('.rec-t');
            const rEl = itemDiv.querySelector('.rec-r');
            if (tEl) tEl.textContent = rec.title;
            if (rEl) rEl.textContent = rec.reason;

            const b = itemDiv.querySelector('.rec-btn');
            if (b) {
              b.addEventListener('click', () => {
                toggleSmartSearchModal(false);
                quickSearch(rec.title);
              });
            }
            recsList.appendChild(itemDiv);
          });
        }

        if (resArea) resArea.style.display = 'block';
      } catch (err) {
        if (errBox) {
          errBox.textContent = err.message || 'Erreur lors de la recherche IA';
          errBox.style.display = 'block';
        }
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Analyser'; }
      }
    }

    function applySmartSearchAndClose() {
      const q = lastSmartSearchKeywords.length > 0 ? lastSmartSearchKeywords[0] : (document.getElementById('smartSearchInput')?.value || '');
      toggleSmartSearchModal(false);
      document.getElementById('queryInput').value = q;
      runSearch();
    }

    function toggleChatModal(show) {
      const modal = document.getElementById('chatModal');
      if (modal) modal.style.display = show ? 'flex' : 'none';
      if (show) {
        setTimeout(() => document.getElementById('chatInput')?.focus(), 100);
      }
    }

    function sendQuickPrompt(promptText) {
      const input = document.getElementById('chatInput');
      if (input) input.value = promptText;
      sendChatMessage();
    }

    async function sendChatMessage() {
      const input = document.getElementById('chatInput');
      const btn = document.getElementById('sendChatBtn');
      const messagesContainer = document.getElementById('chatMessages');

      const text = (input?.value || '').trim();
      if (!text) return;

      input.value = '';
      if (btn) { btn.disabled = true; btn.textContent = '...'; }

      // Afficher le message utilisateur
      const userDiv = document.createElement('div');
      userDiv.style.cssText = 'background: #1f6feb; color: #fff; border-radius: 8px; padding: 10px 14px; max-width: 80%; align-self: flex-end; font-size: 13px; line-height: 1.4;';
      userDiv.textContent = text;
      messagesContainer.appendChild(userDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;

      // Indicator de frappe Gemini
      const typingDiv = document.createElement('div');
      typingDiv.id = 'geminiTypingIndicator';
      typingDiv.style.cssText = 'background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; max-width: 80%; align-self: flex-start; font-size: 12px; color: #8b949e;';
      typingDiv.textContent = 'Gemini 3.8 Flash réfléchit...';
      messagesContainer.appendChild(typingDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;

      chatConversationHistory.push({ role: 'user', content: text });

      try {
        const res = await fetch('/api/ai/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            history: chatConversationHistory.slice(-8)
          })
        });
        const data = await res.json();
        typingDiv.remove();

        const reply = data.reply || "Désolé, je n'ai pas pu traiter votre demande.";
        chatConversationHistory.push({ role: 'model', content: reply });

        const modelDiv = document.createElement('div');
        modelDiv.style.cssText = 'background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; max-width: 85%; align-self: flex-start; font-size: 13px; line-height: 1.5; color: #c9d1d9;';
        
        function formatChatMarkdown(raw) {
          const newline = String.fromCharCode(10);
          const backtick = String.fromCharCode(96);
          const lines = (raw || '').split(newline);
          const html = [];
          for (let line of lines) {
            let l = line
              .split('&').join('&amp;')
              .split('<').join('&lt;')
              .split('>').join('&gt;');
            const parts = l.split('**');
            for (let i = 1; i < parts.length; i += 2) {
              parts[i] = '<strong>' + parts[i] + '</strong>';
            }
            l = parts.join('');
            const cparts = l.split(backtick);
            for (let i = 1; i < cparts.length; i += 2) {
              cparts[i] = '<code style="background: #0d1117; padding: 2px 4px; border-radius: 4px; color: #58a6ff;">' + cparts[i] + '</code>';
            }
            l = cparts.join('');
            const trimmed = l.trimStart();
            if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
              l = '• ' + trimmed.substring(2);
            }
            html.push(l);
          }
          return html.join('<br>');
        }

        modelDiv.innerHTML = formatChatMarkdown(reply);
        messagesContainer.appendChild(modelDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      } catch (err) {
        typingDiv.remove();
        const errDiv = document.createElement('div');
        errDiv.style.cssText = 'background: #3d1b22; border: 1px solid #da3633; border-radius: 8px; padding: 10px 14px; max-width: 80%; align-self: flex-start; font-size: 12px; color: #ff7b72;';
        errDiv.textContent = 'Erreur : ' + (err.message || 'Impossible de contacter Gemini');
        messagesContainer.appendChild(errDiv);
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Envoyer'; }
      }
    }

    function toggleEnrichModal(show) {
      const modal = document.getElementById('enrichModal');
      if (modal) modal.style.display = show ? 'flex' : 'none';
    }

    async function showAnimeMetadataModal(seriesName) {
      if (!seriesName) return;
      toggleEnrichModal(true);

      const titleEl = document.getElementById('enrichCanonicalTitle');
      const origEl = document.getElementById('enrichOriginalTitle');
      const loadingEl = document.getElementById('enrichLoading');
      const contentEl = document.getElementById('enrichContent');

      if (titleEl) titleEl.textContent = seriesName;
      if (origEl) origEl.textContent = "Recherche des données de l'anime...";
      if (loadingEl) { loadingEl.style.display = 'block'; loadingEl.textContent = 'Génération de la fiche encyclopédique par Gemini IA...'; }
      if (contentEl) contentEl.style.display = 'none';

      try {
        const res = await fetch('/api/ai/enrich', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ series_name: seriesName })
        });
        const data = await res.json();
        const meta = data.metadata;

        if (!meta) throw new Error('Aucune information trouvée');

        if (titleEl) titleEl.textContent = meta.canonical_title || seriesName;
        if (origEl) origEl.textContent = meta.original_title ? 'Titre VO : ' + meta.original_title : '';
        document.getElementById('enrichYear').textContent = 'Année ' + (meta.release_year || 'N/A');
        document.getElementById('enrichStudio').textContent = 'Studio : ' + (meta.studio || 'N/A');
        document.getElementById('enrichRating').textContent = '★ ' + (meta.rating || '8.5/10');
        document.getElementById('enrichEpisodesCount').textContent = (meta.total_episodes || 24) + ' épisodes officiels';

        const genresDiv = document.getElementById('enrichGenres');
        if (genresDiv) {
          genresDiv.innerHTML = '';
          (meta.genres || ['Anime']).forEach(g => {
            const sp = document.createElement('span');
            sp.style.cssText = 'background: #21262d; border: 1px solid #30363d; color: #58a6ff; font-size: 11px; padding: 2px 7px; border-radius: 4px;';
            sp.textContent = g;
            genresDiv.appendChild(sp);
          });
        }

        document.getElementById('enrichSynopsis').textContent = meta.synopsis || 'Aucun synopsis disponible.';
        document.getElementById('enrichWatchOrder').textContent = meta.watch_order_tips || "Suivre les saisons dans l'ordre chronologique standard.";

        if (loadingEl) loadingEl.style.display = 'none';
        if (contentEl) contentEl.style.display = 'block';
      } catch (err) {
        if (loadingEl) {
          loadingEl.textContent = 'Impossible de charger la fiche encyclopédique (' + (err.message || 'erreur') + ')';
        }
      }
    }

    function quickSearch(title, channelId) {
      document.getElementById('queryInput').value = title;
      if (channelId) {
        const select = document.getElementById('channelSelect');
        if (select) {
          select.value = channelId;
          onChannelSelected(channelId);
        }
      }
      runSearch();
    }

    function toggleVisionModal(show) {
      const modal = document.getElementById('visionModal');
      if (modal) modal.style.display = show ? 'flex' : 'none';
      if (show && !currentVisionBase64) {
        loadSamplePoster('death_note');
      }
    }

    function onPosterFileSelected(event) {
      const file = event.target?.files?.[0];
      if (!file) return;
      currentVisionMime = file.type || 'image/jpeg';
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result;
        if (typeof dataUrl === 'string') {
          currentVisionBase64 = dataUrl.split(',')[1];
          const previewImg = document.getElementById('posterPreviewImg');
          const previewContainer = document.getElementById('posterPreviewContainer');
          if (previewImg && previewContainer) {
            previewImg.src = dataUrl;
            previewContainer.style.display = 'block';
          }
        }
      };
      reader.readAsDataURL(file);
    }

    function loadSamplePoster(key) {
      const samples = {
        death_note: {
          name: "Death Note",
          caption: "Affiche officielle Death Note - L vs Kira",
          filename: "01.mp4",
          svg: '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400"><rect width="300" height="400" fill="#0d1117"/><rect x="10" y="10" width="280" height="380" fill="#161b22" stroke="#da3633" stroke-width="2" rx="8"/><text x="150" y="140" fill="#fff" font-family="sans-serif" font-size="28" font-weight="bold" text-anchor="middle">DEATH NOTE</text><text x="150" y="180" fill="#da3633" font-family="sans-serif" font-size="16" text-anchor="middle">デスノート - Ryuk &amp; Light</text><rect x="40" y="220" width="220" height="32" fill="#21262d" rx="6"/><text x="150" y="242" fill="#7ee787" font-family="sans-serif" font-size="13" text-anchor="middle">Saison 1 - 37 Episodes</text><text x="150" y="320" fill="#8b949e" font-family="sans-serif" font-size="12" text-anchor="middle">Thriller Psychologique</text></svg>'
        },
        jujutsu: {
          name: "Jujutsu Kaisen",
          caption: "Affiche Shibuya Incident - Gojo Satoru",
          filename: "ep_01.mkv",
          svg: '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400"><rect width="300" height="400" fill="#0a0e14"/><rect x="10" y="10" width="280" height="380" fill="#11161d" stroke="#1f6feb" stroke-width="2" rx="8"/><text x="150" y="140" fill="#58a6ff" font-family="sans-serif" font-size="24" font-weight="bold" text-anchor="middle">JUJUTSU KAISEN</text><text x="150" y="180" fill="#f85149" font-family="sans-serif" font-size="15" text-anchor="middle">Shibuya Incident Arc</text><rect x="40" y="220" width="220" height="32" fill="#21262d" rx="6"/><text x="150" y="242" fill="#7ee787" font-family="sans-serif" font-size="13" text-anchor="middle">Saison 2 - Action</text><text x="150" y="320" fill="#8b949e" font-family="sans-serif" font-size="12" text-anchor="middle">Exorcisme &amp; Surnaturel</text></svg>'
        },
        solo_leveling: {
          name: "Solo Leveling",
          caption: "Solo Leveling Arise - Sung Jin-Woo",
          filename: "01.mp4",
          svg: '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400"><rect width="300" height="400" fill="#0d1117"/><rect x="10" y="10" width="280" height="380" fill="#161b22" stroke="#8957e5" stroke-width="2" rx="8"/><text x="150" y="140" fill="#a371f7" font-family="sans-serif" font-size="24" font-weight="bold" text-anchor="middle">SOLO LEVELING</text><text x="150" y="180" fill="#388bfd" font-family="sans-serif" font-size="15" text-anchor="middle">Shadow Monarch Arise</text><rect x="40" y="220" width="220" height="32" fill="#21262d" rx="6"/><text x="150" y="242" fill="#7ee787" font-family="sans-serif" font-size="13" text-anchor="middle">Saison 1 - Cour 1</text><text x="150" y="320" fill="#8b949e" font-family="sans-serif" font-size="12" text-anchor="middle">Action &amp; Dark Fantasy</text></svg>'
        }
      };

      const sample = samples[key] || samples.death_note;
      const base64 = btoa(unescape(encodeURIComponent(sample.svg)));
      currentVisionBase64 = base64;
      currentVisionMime = 'image/svg+xml';

      const fileInput = document.getElementById('visionFileNameHint');
      const captionInput = document.getElementById('visionCaptionHint');
      const previewImg = document.getElementById('posterPreviewImg');
      const previewContainer = document.getElementById('posterPreviewContainer');

      if (fileInput) fileInput.value = sample.filename;
      if (captionInput) captionInput.value = sample.caption;
      if (previewImg && previewContainer) {
        previewImg.src = 'data:image/svg+xml;base64,' + base64;
        previewContainer.style.display = 'block';
      }
    }

    async function runPosterIdentification() {
      const errBox = document.getElementById('visionErrorBox');
      const resBox = document.getElementById('visionResultBox');
      const btn = document.getElementById('runVisionBtn');

      if (errBox) errBox.style.display = 'none';
      if (resBox) resBox.style.display = 'none';

      if (!currentVisionBase64) {
        if (errBox) {
          errBox.textContent = "Veuillez d'abord choisir une image ou cliquer sur un exemple ci-dessus.";
          errBox.style.display = 'block';
        }
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Analyse Vision en cours...';

      try {
        const payload = {
          image_base64: currentVisionBase64,
          mime_type: currentVisionMime,
          filename_hint: (document.getElementById('visionFileNameHint')?.value || '').trim(),
          caption_hint: (document.getElementById('visionCaptionHint')?.value || '').trim()
        };

        const res = await fetch('/api/ai/identify-poster', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Erreur lors de l'analyse");
        }

        const r = data.result || {};
        document.getElementById('visionAnimeTitle').textContent = '🎬 ' + (r.detected_series || 'Anime détecté');
        document.getElementById('visionConfidence').textContent = (r.confidence || 95) + '% confiance';
        document.getElementById('visionAnimeDetails').textContent = 'Saison ' + (r.season || 1) + (r.episode ? ' • Épisode ' + r.episode : '') + (r.release_year ? ' (' + r.release_year + ')' : '');
        document.getElementById('visionAnimeSynopsis').textContent = r.synopsis || "Titre identifié avec succès à partir des éléments visuels de l'affiche.";

        const genresEl = document.getElementById('visionAnimeGenres');
        if (genresEl) {
          genresEl.innerHTML = '';
          (r.genres || ['Anime', 'Série']).forEach(g => {
            const sp = document.createElement('span');
            sp.style.cssText = 'background: #21262d; border: 1px solid #30363d; color: #58a6ff; font-size: 11px; padding: 2px 6px; border-radius: 4px;';
            sp.textContent = g;
            genresEl.appendChild(sp);
          });
        }

        if (resBox) resBox.style.display = 'block';
      } catch (err) {
        if (errBox) {
          errBox.textContent = err.message || "Erreur d'identification";
          errBox.style.display = 'block';
        }
      } finally {
        btn.disabled = false;
        btn.textContent = "🔍 Identifier l'Anime";
      }
    }

    let currentSearchData = null;
    let currentSelectedSeason = 'all';
    let currentSortAsc = true;

    function toggleSortOrder() {
      currentSortAsc = !currentSortAsc;
      const btn = document.getElementById('sortToggleBtn');
      if (btn) {
        btn.textContent = currentSortAsc ? '↕️ Tri : Épisode 1 → Fin' : '↕️ Tri : Dernier → Épisode 1';
      }
      if (currentSearchData) {
        renderFilteredEpisodes();
      }
    }

    function selectSeason(seasonVal) {
      currentSelectedSeason = seasonVal;
      document.querySelectorAll('.moviebox-season-tab').forEach(t => t.classList.remove('active'));
      const activeTab = document.getElementById('season-tab-' + seasonVal);
      if (activeTab) activeTab.classList.add('active');
      renderFilteredEpisodes();
    }

    function jumpToEpisode(epNum) {
      const epEl = document.getElementById('ep-card-' + epNum);
      if (epEl) {
        epEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        epEl.style.transition = 'all 0.3s';
        epEl.style.boxShadow = '0 0 0 2px #3fb950';
        setTimeout(() => { epEl.style.boxShadow = ''; }, 1500);
        const playBtn = epEl.querySelector('.play-btn');
        if (playBtn) playBtn.click();
      }
    }

    function renderMovieBoxHero(info) {
      const hero = document.getElementById('movieboxHero');
      if (!hero) return;
      if (!info || !info.title || info.title === 'Anime introuvable') {
        hero.style.display = 'none';
        return;
      }

      hero.style.display = 'flex';
      const poster = info.cover || 'https://images.unsplash.com/photo-1578632767115-351597cf2477?w=300&auto=format&fit=crop&q=80';
      const genresHtml = (info.genres || []).map(g => '<span style="background: #21262d; border: 1px solid #30363d; color: #8b949e; padding: 2px 6px; border-radius: 4px; font-size: 11px;">' + g + '</span>').join(' ');

      hero.innerHTML = \`
        <img src="\${poster}" class="moviebox-poster" alt="\${info.title}" onerror="this.src='https://images.unsplash.com/photo-1578632767115-351597cf2477?w=300&auto=format&fit=crop&q=80'" />
        <div class="moviebox-info">
          <h3 class="moviebox-title">\${info.title}</h3>
          <div class="moviebox-meta">
            \${info.score ? \`<span class="moviebox-score">⭐ \${info.score}</span>\` : ''}
            \${info.year ? \`<span class="moviebox-year">📅 \${info.year}</span>\` : ''}
            \${info.total_episodes_official ? \`<span class="moviebox-eps-badge">🎬 \${info.total_episodes_official} épisodes officiels</span>\` : ''}
            \${genresHtml}
          </div>
          <div class="moviebox-synopsis" id="heroSynopsis">\${info.synopsis || 'Pas de synopsis disponible.'}</div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <button type="button" onclick="showAnimeMetadataModal('\${encodeURIComponent(info.title)}')" style="background: #238636; border: none; color: #fff; font-size: 12px; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-weight: 500;">✨ Fiche IA Complète</button>
            <button type="button" onclick="showWatchOrderModal('\${encodeURIComponent(info.title)}')" style="background: #1f6feb; border: none; color: #fff; font-size: 12px; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-weight: 500;">🧭 Ordre de Visionnage IA</button>
          </div>
        </div>
      \`;
    }

    function renderSeasonTabs(seasonsList, totalEpisodesCount) {
      const bar = document.getElementById('movieboxSeasonsBar');
      if (!bar) return;
      if (!seasonsList || seasonsList.length === 0) {
        bar.style.display = 'none';
        return;
      }

      bar.style.display = 'flex';
      let html = \`<button type="button" id="season-tab-all" class="moviebox-season-tab \${currentSelectedSeason === 'all' ? 'active' : ''}" onclick="selectSeason('all')">Toutes les Saisons (\${totalEpisodesCount})</button>\`;

      seasonsList.forEach(s => {
        const isActive = currentSelectedSeason === String(s.season);
        html += \`<button type="button" id="season-tab-\${s.season}" class="moviebox-season-tab \${isActive ? 'active' : ''}" onclick="selectSeason('\${s.season}')">\${s.name} (\${s.episodes_count})</button>\`;
      });

      bar.innerHTML = html;
    }

    function renderFilteredEpisodes() {
      if (!currentSearchData) return;
      const data = currentSearchData;
      const isDedup = document.getElementById('dedupToggle')?.checked ?? true;
      const useDedupList = isDedup && data.deduplicated_episodes && data.deduplicated_episodes.length > 0;
      let rawItems = useDedupList ? [...data.deduplicated_episodes] : [...(data.results || data.episodes || [])];

      // Filtrer par saison sélectionnée
      if (currentSelectedSeason !== 'all') {
        const sNum = parseInt(currentSelectedSeason, 10);
        rawItems = rawItems.filter(item => {
          if (item.season_number !== null && item.season_number !== undefined) {
            return item.season_number === sNum;
          }
          return sNum === 1;
        });
      }

      // Appliquer le sens du tri (Ascendant 1 → N ou Descendant N → 1)
      if (!currentSortAsc) {
        rawItems.reverse();
      }

      document.getElementById('epCount').textContent = rawItems.length;

      // Grille d'accès rapide aux épisodes (Quick Grid)
      const grid = document.getElementById('movieboxQuickGrid');
      if (grid) {
        const distinctEpNumbers = [];
        rawItems.forEach(it => {
          const num = it.episode_number ?? (it.season_number ? it.season_number : null);
          if (num !== null && !distinctEpNumbers.includes(num)) {
            distinctEpNumbers.push(num);
          }
        });

        if (distinctEpNumbers.length > 1) {
          grid.style.display = 'flex';
          grid.innerHTML = '<span style="font-size: 11px; color: #8b949e; align-self: center; margin-right: 6px;">Épisode rapide :</span>' +
            distinctEpNumbers.map(n => \`<button type="button" class="moviebox-ep-btn" onclick="jumpToEpisode(\${n})">\${n}</button>\`).join('');
        } else {
          grid.style.display = 'none';
        }
      }

      const epList = document.getElementById('epList');
      epList.innerHTML = '';

      if (rawItems.length === 0) {
        epList.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-muted);">Aucun épisode pour cette sélection.</div>';
        return;
      }

      rawItems.forEach(item => {
        const div = document.createElement('div');
        div.className = 'episode-item';
        if (item.episode_number) {
          div.id = 'ep-card-' + item.episode_number;
        }

        if (useDedupList) {
          const primary = item.primary_file || {};
          const variants = item.variants || [];
          const hasMultipleVariants = variants.length > 1;

          const isMangaDoc = (item.clean_title || '').includes('Tome') || (item.clean_title || '').includes('Chapitre') || primary.media_type === 'document';
          const typeTag = isMangaDoc ? '<span class="tag-type tag-doc">SCAN MANGA</span>' : '<span class="tag-type tag-video">VIDÉO</span>';
          const epTag = item.season_number
            ? \`S\${String(item.season_number).padStart(2, '0')}E\${String(item.episode_number || 1).padStart(2, '0')}\`
            : \`EP \${String(item.episode_number || 1).padStart(2, '0')}\`;

          div.innerHTML = \`
            <div style="flex: 1; min-width: 0;">
              <div class="episode-header">
                \${typeTag}
                <span class="tag-ep">\${epTag}</span>
                <strong style="color: #fff; font-size: 14px;">\${item.clean_title || item.series_name}</strong>
                \${item.best_quality ? \`<span class="tag-quality">\${item.best_quality}</span>\` : ''}
                \${item.language ? \`<span class="tag-lang">\${item.language}</span>\` : ''}
                \${hasMultipleVariants ? \`<span class="tag-variants">✨ \${variants.length} qualités fusionnées</span>\` : ''}
                <span style="color: var(--text-muted); font-size: 12px; margin-left: 4px;">\${primary.size_mb || 0} MB</span>
              </div>
              <div class="raw-info" title="Fichiers Telegram sources">
                📦 Original Telegram : \${primary.file_name || item.canonical_id}
              </div>
              \${hasMultipleVariants ? \`
                <div style="margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center;">
                  <span style="font-size: 11px; color: #8b949e;">Choisir la qualité :</span>
                  \${variants.map((v, idx) => \`
                    <button type="button" class="variant-btn" data-variant-url="\${v.stream_url}" data-variant-title="\${item.clean_title} (\${v.quality})" data-variant-size="\${v.size_mb}">
                      🎞️ \${v.quality} (\${v.size_mb} MB)
                    </button>
                  \`).join('')}
                </div>
              \` : ''}
            </div>
            <div style="display: flex; align-items: center; margin-left: 12px; gap: 8px;">
              <button class="info-btn" type="button" style="padding: 6px 10px; font-size: 12px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #7ee787; cursor: pointer;" title="Fiche détaillée et synopsis IA">✨ Fiche IA</button>
              \${!isMangaDoc ? \`<button class="play-btn" type="button" style="padding: 6px 14px; font-size: 12px; background: #1f6feb; border-radius: 6px; border: none; color: #fff; cursor: pointer; font-weight: 500;">▶️ Lire</button>\` : \`<a href="\${primary.stream_url}" target="_blank" class="play-btn" style="padding: 6px 14px; font-size: 12px; background: #238636; border-radius: 6px; border: none; color: #fff; text-decoration: none; font-weight: 500;">📖 Ouvrir Scan</a>\`}
              <a href="\${primary.download_url}" class="episode-link" style="padding: 6px 12px; font-size: 12px; border-radius: 6px;" download="\${(item.clean_title || 'media').replace(/[/\\\\?%*:|<>]/g, '_')}.\${isMangaDoc ? 'pdf' : 'mp4'}">📥 Télécharger</a>
            </div>
          \`;

          const infoBtn = div.querySelector('.info-btn');
          if (infoBtn) {
            infoBtn.addEventListener('click', () => {
              showAnimeMetadataModal(item.series_name || item.clean_title);
            });
          }

          const playBtn = div.querySelector('.play-btn');
          if (playBtn && !isMangaDoc) {
            playBtn.addEventListener('click', () => {
              playMedia({
                title: item.clean_title,
                size_mb: primary.size_mb,
                download_url: primary.download_url,
                stream_url: primary.stream_url,
                media_type: 'video',
                file_name: primary.file_name
              });
            });
          }

          div.querySelectorAll('.variant-btn').forEach(btnEl => {
            btnEl.addEventListener('click', (e) => {
              const target = e.currentTarget;
              const vUrl = target.getAttribute('data-variant-url');
              const vTitle = target.getAttribute('data-variant-title');
              const vSize = target.getAttribute('data-variant-size');
              playMedia({
                title: vTitle,
                size_mb: vSize,
                download_url: vUrl + '?dl=1',
                stream_url: vUrl,
                media_type: 'video',
                file_name: vTitle
              });
            });
          });
        } else {
          // Mode Fichier individuel standard
          let typeClass = 'tag-file';
          let typeLabel = 'FICHIER';
          if (item.media_type === 'video') { typeClass = 'tag-video'; typeLabel = 'VIDÉO'; }
          else if (item.media_type === 'audio') { typeClass = 'tag-audio'; typeLabel = 'AUDIO'; }
          else if (item.media_type === 'document') { typeClass = 'tag-doc'; typeLabel = 'DOC'; }

          const canPlay = item.media_type === 'video' || item.media_type === 'audio';

          div.innerHTML = \`
            <div style="flex: 1; min-width: 0;">
              <div class="episode-header">
                <span class="tag-type \${typeClass}">\${typeLabel}</span>
                \${item.episode_number !== null ? \`<span class="tag-ep">EP \${item.episode_number < 10 ? '0' + item.episode_number : item.episode_number}</span>\` : ''}
                <strong style="color: #fff; font-size: 14px;">\${item.title || item.clean_title || item.file_name}</strong>
                \${item.quality ? \`<span class="tag-quality">\${item.quality}</span>\` : ''}
                \${item.language ? \`<span class="tag-lang">\${item.language}</span>\` : ''}
                <span style="color: var(--text-muted); font-size: 12px; margin-left: 4px;">\${item.size_mb} MB</span>
              </div>
              <div class="raw-info" title="Fichier Telegram original">
                📦 Telegram : \${item.file_name}
              </div>
            </div>
            <div style="display: flex; align-items: center; margin-left: 12px; gap: 8px;">
              <button class="info-btn" type="button" style="padding: 6px 10px; font-size: 12px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #7ee787; cursor: pointer;" title="Fiche détaillée et synopsis IA">✨ Fiche IA</button>
              \${canPlay ? \`<button class="play-btn" type="button" style="padding: 6px 14px; font-size: 12px; background: #1f6feb; border-radius: 6px; border: none; color: #fff; cursor: pointer; font-weight: 500;">▶️ Lire</button>\` : ''}
              <a href="\${item.download_url}" class="episode-link" style="padding: 6px 12px; font-size: 12px; border-radius: 6px;" download="\${(item.file_name || 'media').replace(/[/\\\\?%*:|<>]/g, '_')}">📥 Télécharger</a>
            </div>
          \`;

          const infoBtn = div.querySelector('.info-btn');
          if (infoBtn) {
            infoBtn.addEventListener('click', () => {
              showAnimeMetadataModal(item.clean_title || item.title || item.file_name);
            });
          }

          const playBtn = div.querySelector('.play-btn');
          if (playBtn) {
            playBtn.addEventListener('click', () => playMedia(item));
          }
        }

        epList.appendChild(div);
      });
    }

    function renderPagination(data) {
      const pBar = document.getElementById('paginationBar');
      if (!pBar) return;
      if (!data || (!data.has_next && !data.has_prev && !data.has_more_telegram)) {
        pBar.style.display = 'none';
        return;
      }

      pBar.style.display = 'flex';
      const currentPage = data.page || 1;
      const totalPages = data.total_pages || 1;

      pBar.innerHTML = \`
        <div style="font-size: 13px; color: #8b949e;">
          Page <strong style="color: #fff;">\${currentPage}</strong> sur <strong style="color: #fff;">\${totalPages}</strong> (\${data.total_found || 0} fichiers)
        </div>
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
          \${data.has_prev ? \`<button type="button" onclick="changePage(\${currentPage - 1})" style="background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; cursor: pointer;">⬅️ Précédent</button>\` : ''}
          \${data.has_next ? \`<button type="button" onclick="changePage(\${currentPage + 1})" style="background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; cursor: pointer;">Suivant ➡️</button>\` : ''}
          \${data.has_more_telegram && data.next_offset_id ? \`
            <button type="button" onclick="fetchNextTelegramBatch(\${data.next_offset_id})" style="background: #1f6feb; border: none; color: #fff; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 500;">
              ⚡ Charger les anciens épisodes Telegram (\${data.next_offset_id})
            </button>
          \` : ''}
        </div>
      \`;
    }

    async function changePage(newPage) {
      runSearch(newPage);
    }

    async function fetchNextTelegramBatch(offsetId) {
      runSearch(1, offsetId);
    }

    async function runSearch(targetPage = 1, offsetId = null) {
      clearError();
      const q = document.getElementById('queryInput').value.trim();
      const channel = document.getElementById('channelInput').value.trim();
      const btn = document.getElementById('searchBtn');
      const isDedup = document.getElementById('dedupToggle')?.checked ?? true;

      btn.disabled = true;
      btn.textContent = 'Chargement...';

      try {
        let url = '/search?q=' + encodeURIComponent(q) + '&channel=' + encodeURIComponent(channel) + '&type=' + encodeURIComponent(currentFilterType) + '&dedup=' + (isDedup ? 'true' : 'false') + '&page=' + targetPage;
        if (offsetId) {
          url += '&offset_id=' + offsetId;
        }

        const res = await fetch(url);
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || 'Erreur réseau (' + res.status + ')');
        }
        const data = await res.json();
        currentSearchData = data;
        currentSelectedSeason = 'all';

        // Mettre à jour le badge de statistiques de déduplication
        const statsBadge = document.getElementById('dedupStatsBadge');
        if (statsBadge) {
          if (isDedup && (data.duplicates_eliminated > 0 || (data.deduplicated_episodes && data.deduplicated_episodes.length > 0))) {
            statsBadge.textContent = '✨ ' + (data.duplicates_eliminated || 0) + ' doublons fusionnés • Tri ordonné S01E01 → ' + (data.deduplicated_episodes?.length ? 'E' + String(data.deduplicated_episodes.length).padStart(2, '0') : '');
            statsBadge.style.display = 'inline-block';
          } else {
            statsBadge.style.display = 'none';
          }
        }

        document.getElementById('resultsArea').style.display = 'block';

        const catLabels = { all: 'Tous types', video: 'Vidéos & Films', audio: 'Musique & Audio', document: 'Documents & Scans', archive: 'Archives' };
        document.getElementById('categoryLabel').textContent = catLabels[currentFilterType] || 'Tous types';

        // 1. Rendre le Banner MovieBox Hero
        renderMovieBoxHero(data.anime_info);

        // 2. Rendre les Onglets de Saisons
        renderSeasonTabs(data.seasons_list, data.total_found || (data.deduplicated_episodes || []).length);

        // 3. Rendre la Liste d'épisodes et la Grille rapide
        renderFilteredEpisodes();

        // 4. Rendre la Barre de pagination et pagination Telegram
        renderPagination(data);

      } catch (err) {
        showError('Erreur lors de la recherche: ' + (err.message || err));
      } finally {
        btn.disabled = false;
        btn.textContent = 'Explorer';
      }
    }

    function playMedia(item) {
      clearError();
      const section = document.getElementById('playerSection');
      const videoPlayer = document.getElementById('videoPlayer');
      const audioContainer = document.getElementById('audioContainer');
      const audioPlayer = document.getElementById('audioPlayer');
      const titleEl = document.getElementById('nowPlayingTitle');
      const dlBtn = document.getElementById('nowPlayingDownloadBtn');

      if (!section) return;
      section.style.display = 'block';

      titleEl.textContent = (item.title || item.clean_title || item.file_name) + ' (' + item.size_mb + ' MB)';
      dlBtn.href = item.download_url;
      dlBtn.setAttribute('download', (item.file_name || 'media').replace(/[/\\?%*:|<>]/g, '_'));
      dlBtn.style.display = 'inline-block';

      videoPlayer.onerror = () => {
        showError("Impossible de décoder cette vidéo directement dans le lecteur web (format MKV ou codec non pris en charge). Utilisez le bouton de téléchargement pour la lire avec VLC.");
      };
      audioPlayer.onerror = () => {
        showError("Impossible de lire ce format audio dans le navigateur. Utilisez le bouton de téléchargement pour l'écouter avec votre lecteur.");
      };

      if (item.media_type === 'audio') {
        videoPlayer.pause();
        videoPlayer.style.display = 'none';

        audioContainer.style.display = 'block';
        document.getElementById('audioTrackName').textContent = item.track_title || item.clean_title || item.file_name;
        document.getElementById('audioTrackArtist').textContent = item.artist ? 'Artiste : ' + item.artist : 'Canal Telegram';

        audioPlayer.src = item.stream_url;
        audioPlayer.load();
        audioPlayer.play().catch(e => console.log('Audio play need interaction:', e));
      } else {
        audioPlayer.pause();
        audioContainer.style.display = 'none';

        videoPlayer.style.display = 'block';
        videoPlayer.src = item.stream_url;
        videoPlayer.load();
        videoPlayer.play().catch(e => console.log('Video play need interaction:', e));
      }

      section.scrollIntoView({ behavior: 'smooth' });
    }

    function onChannelSelected(val) {
      const customInput = document.getElementById('channelInput');
      if (val === 'custom') {
        customInput.style.display = 'inline-block';
        customInput.value = '';
        customInput.focus();
      } else {
        customInput.style.display = 'none';
        customInput.value = val;
        runSearch();
      }
    }

    function toggleSessionModal(show) {
      const modal = document.getElementById('sessionModal');
      const msg = document.getElementById('sessionStatusMsg');
      if (modal) {
        modal.style.display = show ? 'flex' : 'none';
        if (show && msg) msg.style.display = 'none';
      }
    }

    async function retryTelegramConnection() {
      const accountEl = document.getElementById('tgAccountText');
      const detailsEl = document.getElementById('tgAccountDetails');
      const dotEl = document.getElementById('tgAccountDot');
      if (accountEl) accountEl.textContent = 'Telegram MTProto : Reconnexion en cours...';
      if (detailsEl) detailsEl.textContent = 'Test du client MTProto...';
      if (dotEl) {
        dotEl.style.background = '#e3b341';
        dotEl.style.boxShadow = '0 0 8px #e3b341';
      }

      try {
        const res = await fetch('/api/telegram/reconnect', { method: 'POST' });
        const data = await res.json();
        await loadTelegramStatus();
      } catch (err) {
        console.error('Erreur reconnexion:', err);
      }
    }

    async function saveNewSession() {
      const input = document.getElementById('newSessionInput');
      const msg = document.getElementById('sessionStatusMsg');
      const btn = document.getElementById('saveSessionBtn');
      const sessionVal = (input ? input.value : '').trim();

      if (!sessionVal) {
        if (msg) {
          msg.style.display = 'block';
          msg.style.color = '#ff7b72';
          msg.textContent = 'Veuillez coller une SESSION_STRING.';
        }
        return;
      }

      if (btn) {
        btn.disabled = true;
        btn.textContent = 'Connexion en cours...';
      }
      if (msg) {
        msg.style.display = 'block';
        msg.style.color = '#58a6ff';
        msg.textContent = 'Test de la nouvelle session avec Telegram...';
      }

      try {
        const res = await fetch('/api/telegram/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_string: sessionVal })
        });
        const data = await res.json();
        if (data.success) {
          if (msg) {
            msg.style.color = '#3fb950';
            msg.textContent = '✅ Connecté avec succès ! Fermeture...';
          }
          setTimeout(() => {
            toggleSessionModal(false);
            if (input) input.value = '';
            loadTelegramStatus();
            runSearch();
          }, 1200);
        } else {
          if (msg) {
            msg.style.color = '#ff7b72';
            msg.textContent = '❌ ' + (data.status?.error?.message || data.error || 'Échec de connexion avec cette session.');
          }
        }
      } catch (err) {
        if (msg) {
          msg.style.color = '#ff7b72';
          msg.textContent = 'Erreur réseau lors de la mise à jour.';
        }
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.textContent = 'Tester & Connecter';
        }
      }
    }

    async function loadTelegramStatus() {
      try {
        const res = await fetch('/channels');
        if (!res.ok) return;
        const data = await res.json();
        const accountEl = document.getElementById('tgAccountText');
        const detailsEl = document.getElementById('tgAccountDetails');
        const dotEl = document.getElementById('tgAccountDot');
        const alertEl = document.getElementById('tgConflictAlert');

        if (data.connected) {
          if (dotEl) {
            dotEl.style.background = '#3fb950';
            dotEl.style.boxShadow = '0 0 8px #3fb950';
          }
          if (accountEl) {
            accountEl.textContent = 'Telegram MTProto : Connecté (' + (data.user?.username ? '@' + data.user.username : data.user?.firstName || 'Compte') + ')';
          }
          if (detailsEl) {
            detailsEl.textContent = 'Session active • ' + (data.channels?.length || 0) + ' canaux disponibles';
          }
          if (alertEl) {
            alertEl.style.display = 'none';
          }

          const selectEl = document.getElementById('channelSelect');
          if (selectEl && data.channels && data.channels.length > 0) {
            selectEl.innerHTML = '';
            data.channels.forEach(ch => {
              const opt = document.createElement('option');
              opt.value = ch.id;
              opt.textContent = '📁 ' + ch.title + ' (' + ch.id + ')';
              if (ch.id === '-1004272203145' || ch.title === 'NLSbox') {
                opt.selected = true;
              }
              selectEl.appendChild(opt);
            });
            const customOpt = document.createElement('option');
            customOpt.value = 'custom';
            customOpt.textContent = '✏️ Saisir un autre ID ou @canal...';
            selectEl.appendChild(customOpt);

            document.getElementById('channelInput').value = selectEl.value;
          }
        } else {
          const isDuplicated = data.error?.code === 406 || data.error?.type === 'AUTH_KEY_DUPLICATED';
          if (dotEl) {
            dotEl.style.background = isDuplicated ? '#f85149' : '#d29922';
            dotEl.style.boxShadow = isDuplicated ? '0 0 8px #f85149' : '0 0 8px #d29922';
          }
          if (accountEl) {
            accountEl.textContent = isDuplicated 
              ? 'Telegram MTProto : ⚠️ Session Dupliquée (Erreur 406)' 
              : 'Telegram MTProto : Déconnecté';
          }
          if (detailsEl) {
            detailsEl.textContent = isDuplicated 
              ? "Clé révoquée par conflit d'instance" 
              : (data.error?.message || 'Identifiants ou session invalides');
          }
          if (alertEl) {
            alertEl.style.display = isDuplicated ? 'block' : 'none';
          }
        }
      } catch (err) {
        console.warn('Could not load Telegram channels:', err);
      }
    }

    async function checkGeminiStatus() {
      try {
        const res = await fetch('/api/ai/status');
        const data = await res.json();
        const badge = document.getElementById('aiBadge');
        const desc = document.getElementById('aiStatusDesc');
        if (data.available || data.configured) {
          if (badge) {
            badge.style.background = '#238636';
            badge.textContent = 'ACTIF (' + (data.model || 'gemini-3.8-flash') + ')';
          }
          if (desc) {
            desc.textContent = 'Modèle ' + (data.model || 'gemini-3.8-flash') + ' actif • Déduplication S01E01, recherche sémantique, vision & assistant chat';
          }
        } else {
          if (badge) {
            badge.style.background = '#8957e5';
            badge.textContent = 'MODE LOCAL';
          }
          if (desc) {
            desc.textContent = "Déduplication heuristique locale active. Configurez GEMINI_API_KEY pour activer l'IA Gemini 3.8 Flash.";
          }
        }
      } catch (e) {
        console.warn('Could not check Gemini status', e);
      }
    }

    // Auto-run first search and listen to Enter key
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('queryInput')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runSearch();
      });
      document.getElementById('channelInput')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runSearch();
      });
      loadTelegramStatus();
      checkGeminiStatus();
      runSearch();
    });
  </script>
</body>
</html>
    `);
});

// Status & Health endpoints (compatibles Render, Docker, Kubernetes, monitoring)
app.get(["/status", "/health", "/healthz"], (req: Request, res: Response) => {
  return res.status(200).json({
    status: "ok",
    message: "En ligne - Sanitizer & Tri V2 OK",
    app: "NLSbox Backend Pro",
    telegram: {
      connected: tgStatusInfo.connected,
      user: tgStatusInfo.user,
      error: tgStatusInfo.error,
    },
    timestamp: new Date().toISOString(),
  });
});

// Diagnostic & Status Telegram MTProto
app.get("/api/telegram/status", (req: Request, res: Response) => {
  return res.json(tgStatusInfo);
});

// Mettre à jour la session Telegram (résout les erreurs 406 AUTH_KEY_DUPLICATED)
app.post("/api/telegram/session", async (req: Request, res: Response) => {
  const { session_string } = req.body || {};
  if (!session_string || typeof session_string !== "string" || session_string.trim().length < 20) {
    return res.status(400).json({
      success: false,
      error: "SESSION_STRING invalide ou trop courte.",
    });
  }

  const success = await setTelegramSession(session_string.trim());
  return res.json({
    success,
    status: tgStatusInfo,
    message: success
      ? "Nouvelle session Telegram connectée avec succès !"
      : "Échec de connexion avec cette session.",
  });
});

// Forcer la reconnexion Telegram MTProto
app.post("/api/telegram/reconnect", async (req: Request, res: Response) => {
  await closeTelegramClient();
  const client = await getTelegramClient(true);
  return res.json({
    success: !!client,
    status: tgStatusInfo,
  });
});

// --- MODULE IA GEMINI : DÉDUPLICATION, TRI ET RECONNAISSANCE VISUELLE ---

// Statut de disponibilité de l'IA Gemini
app.get("/api/ai/status", (_req: Request, res: Response) => {
  const currentModel = getActiveGeminiModel();
  return res.json({
    available: isGeminiAvailable(),
    model: currentModel,
    features: {
      smart_deduplication: true,
      sequential_ordering: true,
      uniform_renaming: true,
      multimodal_poster_identification: true,
      natural_language_search: true,
      conversational_assistant: true,
      metadata_enrichment: true,
    },
    message: isGeminiAvailable()
      ? `Module Gemini IA actif (${currentModel} : Déduplication, Tri chronologique, Vision d'affiche, Recherche sémantique & Assistant Chat).`
      : "Module Gemini en mode heuristique local. Définissez GEMINI_API_KEY pour débloquer toutes les capacités avancées.",
  });
});

// Endpoint de déduplication et ordonnancement de lots de fichiers pour les applications clientes
app.post("/api/ai/deduplicate", async (req: Request, res: Response) => {
  try {
    const { items, series_hint } = req.body;
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: "Fournissez un tableau 'items' non vide." });
    }
    const deduplicated = await aiDeduplicateAndSortEpisodes(items, series_hint || "");
    return res.json({
      success: true,
      ai_processed: isGeminiAvailable(),
      original_count: items.length,
      deduplicated_count: deduplicated.length,
      duplicates_merged: Math.max(0, items.length - deduplicated.length),
      episodes: deduplicated,
    });
  } catch (err: any) {
    console.error("[API AI Deduplicate] Erreur:", err);
    return res.status(500).json({ error: err.message || "Erreur interne" });
  }
});

// Endpoint de recherche intelligente en langage naturel (Gemini 3.8 Flash)
app.post("/api/ai/smart-search", async (req: Request, res: Response) => {
  try {
    const { query, media_titles } = req.body;
    if (!query || typeof query !== "string") {
      return res.status(400).json({ error: "Fournissez une 'query' textuelle non vide." });
    }
    const result = await aiSmartSearchAssistant(query, Array.isArray(media_titles) ? media_titles : []);
    return res.json({
      success: true,
      ai_processed: isGeminiAvailable(),
      result,
    });
  } catch (err: any) {
    console.error("[API AI Smart Search] Erreur:", err);
    return res.status(500).json({ error: err.message || "Erreur interne" });
  }
});

// Endpoint de l'assistant conversationnel NLSbox (Gemini 3.8 Flash)
app.post("/api/ai/chat", async (req: Request, res: Response) => {
  try {
    const { message, history, media_titles } = req.body;
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Fournissez un 'message' textuel non vide." });
    }
    const reply = await aiChatAssistant(
      message,
      Array.isArray(history) ? history : [],
      Array.isArray(media_titles) ? media_titles : []
    );
    return res.json({
      success: true,
      reply,
    });
  } catch (err: any) {
    console.error("[API AI Chat] Erreur:", err);
    return res.status(500).json({ error: err.message || "Erreur interne" });
  }
});

// Endpoint d'enrichissement de métadonnées (synopsis, genres, studio, ordre de visionnage)
app.post("/api/ai/enrich", async (req: Request, res: Response) => {
  try {
    const { series_name } = req.body;
    if (!series_name || typeof series_name !== "string") {
      return res.status(400).json({ error: "Fournissez un 'series_name' textuel." });
    }
    const metadata = await aiEnrichAnimeMetadata(series_name);
    return res.json({
      success: !!metadata,
      metadata,
    });
  } catch (err: any) {
    console.error("[API AI Enrich] Erreur:", err);
    return res.status(500).json({ error: err.message || "Erreur interne" });
  }
});

// Endpoint de reconnaissance d'anime par image/affiche (Gemini Vision Multimodal)
app.post("/api/ai/identify-poster", async (req: Request, res: Response) => {
  try {
    const { image_base64, mime_type, filename_hint, caption_hint, channel_id, message_id } = req.body;

    let base64Data = image_base64;
    const actualMime = mime_type || "image/jpeg";

    // Si channel_id et message_id sont fournis et que MTProto est connecté, télécharger la vignette Telegram
    if (!base64Data && channel_id && message_id) {
      const client = await getTelegramClient();
      if (client) {
        try {
          const entity = await client.getEntity(channel_id);
          const [msg] = await client.getMessages(entity, { ids: [parseInt(message_id, 10)] });
          if (msg && (msg.media || msg.photo)) {
            const buffer = await client.downloadMedia(msg, { thumb: 0 });
            if (buffer && Buffer.isBuffer(buffer)) {
              base64Data = buffer.toString("base64");
            }
          }
        } catch (tgErr: any) {
          console.warn("[Telegram Vision] Impossible de télécharger la vignette Telegram:", tgErr.message);
        }
      }
    }

    if (!base64Data) {
      return res.status(400).json({
        error: "Aucune image fournie. Envoyez 'image_base64' ou les identifiants 'channel_id' et 'message_id' d'un média Telegram.",
      });
    }

    const result = await aiIdentifyAnimeFromPoster(
      base64Data,
      actualMime,
      filename_hint || "",
      caption_hint || ""
    );

    if (!result) {
      return res.status(422).json({
        error: "Impossible d'identifier l'anime à partir de cette image (clé GEMINI_API_KEY requise ou image non reconnue).",
      });
    }

    return res.json({
      success: true,
      result,
    });
  } catch (err: any) {
    console.error("[API AI Vision] Erreur:", err);
    return res.status(500).json({ error: err.message || "Erreur interne" });
  }
});

// 1. Get user channels and connection status from Telegram MTProto
app.get("/channels", async (req: Request, res: Response) => {
  try {
    const client = await getTelegramClient();
    if (!client) {
      return res.json({
        connected: false,
        error: tgStatusInfo.error,
        channels: [],
      });
    }
    const me = await client.getMe();
    const dialogs = await client.getDialogs({ limit: 30 });
    const channels = dialogs
      .filter((d) => d.isChannel || d.isGroup)
      .map((d) => ({
        id: d.id?.toString(),
        title: d.title,
        username: (d.entity as any)?.username || null,
      }));

    tgStatusInfo.connected = true;
    tgStatusInfo.user = {
      id: me.id?.toString(),
      username: me.username || null,
      firstName: me.firstName || "",
    };
    tgStatusInfo.channelsCount = channels.length;

    return res.json({
      connected: true,
      user: tgStatusInfo.user,
      channels,
      error: null,
    });
  } catch (err: any) {
    console.error("Error fetching Telegram channels:", err);
    return res.status(500).json({ error: err.message, status: tgStatusInfo });
  }
});

// 2. Universal Search endpoint (Telegram files: Video, Music, Movies, Series, Anime, Scans, Documents)
app.get("/search", async (req: Request, res: Response) => {
  try {
    const q = (req.query.q as string || "").trim();
    const channel = (req.query.channel as string || "").trim();
    const rawPage = req.query.page as string;
    const filterType = (req.query.type as string || "all").toLowerCase().trim();

    const pageNum = Math.max(1, parseInt(rawPage, 10) || 1);
    const cacheKey = `${channel}:${q.toLowerCase().trim()}:${filterType}`;

    let metadata: AnimeMetadata;
    let items: any[] = [];

    const cached = getCachedSearch(cacheKey);
    if (cached) {
      metadata = cached.anime_info;
      items = cached.episodes;
    } else {
      let rawItems: any[] = [];
      let isRealTelegram = false;

      const client = await getTelegramClient();

      if (client && channel && channel !== "demo" && channel !== "-1001234567890") {
        try {
          // Paramètres de pagination et recherche Telegram
          const rawOffsetId = parseInt(req.query.offset_id as string, 10);
          const offsetId = (!isNaN(rawOffsetId) && rawOffsetId > 0) ? rawOffsetId : 0;
          
          const rawLimit = parseInt(req.query.limit as string, 10);
          // Si recherche textuelle (ex: "naruto"), on ratisse jusqu'à 500 messages pour ne manquer aucun épisode historique (Épisode 1, 2, 3...)
          const fetchLimit = (!isNaN(rawLimit) && rawLimit > 0)
            ? Math.min(Math.max(10, rawLimit), 1000)
            : (q && q !== "*" ? 500 : 150);

          console.log(`[Telegram Search] Querying "${q || '<all>'}" in channel "${channel}" (limit: ${fetchLimit}, offsetId: ${offsetId})...`);
          const entity = await withTimeout(client.getEntity(channel), 7000, "Timeout accès canal Telegram (7s)");
          
          let messages: any[] = [];
          if (q && q !== "*") {
            const searchOpts: any = { search: q, limit: fetchLimit };
            if (offsetId > 0) searchOpts.offsetId = offsetId;
            messages = await withTimeout(client.getMessages(entity, searchOpts), 8000, "Timeout recherche messages Telegram (8s)");

            // Recherche élargie de secours : si la requête est composée (ex: "naruto vostfr" ou "naruto s01") et retourne peu de résultats
            if (messages.length < 5 && q.includes(" ")) {
              const primaryWord = q.split(/[\s._-]+/)[0].trim();
              if (primaryWord.length >= 3 && primaryWord.toLowerCase() !== q.toLowerCase()) {
                console.log(`[Telegram Search] Recherche élargie sur le terme racine "${primaryWord}"...`);
                try {
                  const broader = await withTimeout(client.getMessages(entity, { search: primaryWord, limit: fetchLimit }), 5000, "Timeout recherche élargie");
                  if (broader.length > messages.length) {
                    messages = broader;
                  }
                } catch (e: any) {
                  console.warn("[Telegram Search] Échec recherche élargie:", e.message);
                }
              }
            }
          } else {
            const histOpts: any = { limit: fetchLimit };
            if (offsetId > 0) histOpts.offsetId = offsetId;
            messages = await withTimeout(client.getMessages(entity, histOpts), 8000, "Timeout historique Telegram (8s)");
          }

          // Déterminer le prochain offset_id pour pagination continue Telegram
          let nextOffsetIdCandidate: number | null = null;
          if (messages.length >= fetchLimit && messages.length > 0) {
            const minMsgId = Math.min(...messages.map((m: any) => m.id));
            if (minMsgId > 1) {
              nextOffsetIdCandidate = minMsgId;
            }
          }
          (req as any)._nextOffsetId = nextOffsetIdCandidate;

          for (const m of messages) {
            const doc = m.media && ("document" in m.media ? (m.media as any).document : "video" in m.media ? (m.media as any).video : null);
            if (doc) {
              const fileNameAttr = doc.attributes?.find((a: any) => a.fileName);
              const audioAttr = doc.attributes?.find((a: any) => a.className === "DocumentAttributeAudio");
              const videoAttr = doc.attributes?.find((a: any) => a.className === "DocumentAttributeVideo");
              
              let fileName = fileNameAttr?.fileName || "";
              if (!fileName && audioAttr && audioAttr.title) {
                fileName = `${audioAttr.performer ? audioAttr.performer + " - " : ""}${audioAttr.title}.mp3`;
              }
              if (!fileName && m.message) {
                fileName = m.message.split("\n")[0].slice(0, 100);
              }
              
              const mimeType = doc.mimeType || (audioAttr ? "audio/mpeg" : videoAttr ? "video/mp4" : "application/octet-stream");
              const sizeBytes = Number(doc.size || 0);
              const sizeMb = parseFloat((sizeBytes / (1024 * 1024)).toFixed(1));

              rawItems.push({
                message_id: m.id,
                channel_id: channel,
                file_name: fileName,
                caption: m.message || "",
                size_mb: sizeMb,
                mime_type: mimeType,
                audio_attr: audioAttr ? { title: audioAttr.title, performer: audioAttr.performer } : null,
                duration: audioAttr?.duration || videoAttr?.duration || null,
              });
            }
          }
          isRealTelegram = true;
          console.log(`[Telegram Search] Found ${rawItems.length} media messages in channel "${channel}"`);
        } catch (err: any) {
          console.warn(`[Telegram Search] Failed to search channel ${channel}:`, err.message);
          return res.status(404).json({
            detail: `Impossible d'accéder au canal "${channel}": ${err.message}.`,
            anime_info: {
              title: q || "Contenu Telegram",
              synopsis: "Erreur d'accès au canal",
              cover: null,
              score: null,
              genres: [],
              year: null,
              total_episodes_official: null,
            },
            results: [],
            episodes: [],
            episodes_found: 0,
          });
        }
      }

      // Si mode démo ou aucun client Telegram
      if (!isRealTelegram) {
        // Jeu de données de démonstration avec variantes typiques Telegram pour tester la déduplication et le renommage
        rawItems.push(
          // Épisode 1 en 1080p
          {
            message_id: 1001,
            channel_id: channel || "-1004272203145",
            file_name: "[Team-Fansub] @AnimeFR_Death_Note_-_01_[1080p_x264_VOSTFR]_t.me_animenz.mp4",
            caption: "Death Note Episode 01 en FULL HD 1080p !",
            size_mb: 320.5,
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1420,
          },
          // Épisode 1 en 720p (doublon même épisode, nom différent)
          {
            message_id: 10011,
            channel_id: channel || "-1004272203145",
            file_name: "Death Note S1 épisode 1 VOSTFR [720p].mkv",
            caption: "Death Note S01E01 version 720p légère",
            size_mb: 180.2,
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1420,
          },
          // Épisode 1 version nom brut mobile
          {
            message_id: 10012,
            channel_id: channel || "-1004272203145",
            file_name: "death note ep1.mp4",
            caption: "Death Note 01",
            size_mb: 120.0,
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1420,
          },
          // Épisode 2 en 1080p
          {
            message_id: 1002,
            channel_id: channel || "-1004272203145",
            file_name: "[Team-Fansub] @AnimeFR_Death_Note_-_02_[1080p_x264_VOSTFR]_t.me_animenz.mp4",
            caption: "Death Note Episode 02 disponible !",
            size_mb: 335.7,
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1420,
          },
          // Épisode 2 en 720p
          {
            message_id: 10021,
            channel_id: channel || "-1004272203145",
            file_name: "Death Note ep 2 720p.mkv",
            caption: "Death Note 02 en 720p",
            size_mb: 190.5,
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1420,
          }
        );

        // Épisodes Death Note 3 à 12
        for (let i = 3; i <= 12; i++) {
          const epNumStr = i < 10 ? `0${i}` : `${i}`;
          const rawFileName = i === 4
            ? "04.mp4"
            : `[Team-Fansub] @AnimeFR_Death_Note_-_${epNumStr}_[1080p_x264_VOSTFR]_t.me_animenz.mp4`;
          const caption = i === 4
            ? "Death Note - Épisode 04 (L et Kira se rapprochent)"
            : `Death Note Episode ${epNumStr} disponible !`;

          rawItems.push({
            message_id: 1000 + i,
            channel_id: channel || "-1004272203145",
            file_name: rawFileName,
            caption,
            size_mb: parseFloat((310.0 + (i % 4) * 18.5).toFixed(1)),
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1420,
          });
        }

        // Épisodes Naruto Saison 1 (Épisode 01 à 24) avec variantes 1080p, 720p, VF et VOSTFR
        for (let i = 1; i <= 24; i++) {
          const epPad = i < 10 ? `0${i}` : `${i}`;
          // Version 1080p VOSTFR principale
          rawItems.push({
            message_id: 2000 + i,
            channel_id: channel || "-1004272203145",
            file_name: `[Erai-raws] @AnimeZone_Naruto_-_${epPad}_[1080p_x264_VOSTFR]_t.me_animenz.mp4`,
            caption: `Naruto Saison 1 - Épisode ${epPad} [1080p VOSTFR]`,
            size_mb: parseFloat((340.0 + (i % 3) * 15.2).toFixed(1)),
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1410,
          });

          // Doublon version 720p pour les 10 premiers épisodes
          if (i <= 10) {
            rawItems.push({
              message_id: 2500 + i,
              channel_id: channel || "-1004272203145",
              file_name: `Naruto S01E${epPad} VOSTFR 720p.mkv`,
              caption: `Naruto E${epPad} 720p léger`,
              size_mb: 175.5,
              mime_type: "video/mp4",
              audio_attr: null,
              duration: 1410,
            });
          }

          // Version VF pour les épisodes 1 à 5
          if (i <= 5) {
            rawItems.push({
              message_id: 2800 + i,
              channel_id: channel || "-1004272203145",
              file_name: `Naruto Épisode ${epPad} VF.mp4`,
              caption: `Naruto 0${i} en version française VF`,
              size_mb: 185.0,
              mime_type: "video/mp4",
              audio_attr: null,
              duration: 1410,
            });
          }
        }

        // Naruto Saison 2 (Épisodes 25 à 40)
        for (let i = 25; i <= 40; i++) {
          const epPad = `${i}`;
          rawItems.push({
            message_id: 2000 + i,
            channel_id: channel || "-1004272203145",
            file_name: `Naruto S02E${i} [1080p VOSTFR].mp4`,
            caption: `Naruto Saison 2 Épisode ${epPad}`,
            size_mb: 350.0,
            mime_type: "video/mp4",
            audio_attr: null,
            duration: 1410,
          });
        }

        // Naruto Scans & Manga (Tomes et Chapitres pour tester la gestion manga)
        for (let t = 1; t <= 5; t++) {
          rawItems.push({
            message_id: 3000 + t,
            channel_id: channel || "-1004272203145",
            file_name: `Naruto - Tome 0${t}.pdf`,
            caption: `Manga Scan Naruto Tome 0${t} FR haute qualité`,
            size_mb: 65.4,
            mime_type: "application/pdf",
            audio_attr: null,
            duration: null,
          });
        }
        for (let c = 1; c <= 10; c++) {
          const cPad = c < 10 ? `0${c}` : `${c}`;
          rawItems.push({
            message_id: 3100 + c,
            channel_id: channel || "-1004272203145",
            file_name: `Naruto - Chapitre ${cPad}.cbz`,
            caption: `Chapitre ${cPad} Naruto scan complet`,
            size_mb: 22.1,
            mime_type: "application/octet-stream",
            audio_attr: null,
            duration: null,
          });
        }
      }

      // Nettoyage intelligent universel (Musique, Vidéos, Films, Séries, Scans, Documents)
      const cleaned = filterAndSortEpisodes(rawItems, q);

      // Si des épisodes avec numérotation d'anime/série sont détectés et qu'une recherche q existe, on tente AniList
      const isAnimeSeries = cleaned.some(c => c.episode_number !== null);
      if (isAnimeSeries && q && q.length > 2) {
        try {
          metadata = await fetchAnimeMetadata(q);
        } catch {
          metadata = {
            title: q,
            synopsis: `Contenu multimédia Telegram NLSbox (${cleaned.length} éléments trouvés).`,
            cover: null,
            banner: null,
            score: null,
            genres: ["Telegram", "NLSbox"],
            year: new Date().getFullYear(),
            total_episodes_official: cleaned.length,
          };
        }
      } else {
        metadata = {
          title: q || "Bibliothèque Multimédia NLSbox",
          synopsis: `Exploration Telegram NLSbox : ${cleaned.length} fichiers répertoriés (vidéos, musique, documents).`,
          cover: null,
          banner: null,
          score: null,
          genres: ["Telegram", "Media", "NLSbox"],
          year: new Date().getFullYear(),
          total_episodes_official: cleaned.length,
        };
      }

      items = cleaned.map(item => {
        const rawFileName = item.file_name || `${item.clean_title || "media"}.${item.media_type === "audio" ? "mp3" : "mp4"}`;
        const cleanSafeName = rawFileName.replace(/[/\\?%*:|"<>]/g, "_").trim();
        const encodedFileName = encodeURIComponent(cleanSafeName);
        const channelParam = encodeURIComponent(item.channel_id || channel || "-1004272203145");

        return {
          ...item,
          title: item.clean_title,
          stream_url: `/download/${channelParam}/${item.message_id}/${encodedFileName}`,
          download_url: `/download/${channelParam}/${item.message_id}/${encodedFileName}?dl=1`,
        };
      });

      if (items.length > 0) {
        setCachedSearch(cacheKey, { anime_info: metadata, episodes: items });
      }
    }

    // Filtrer par type si demandé (vidéo, audio, document, archive)
    let filteredItems = items;
    if (filterType && filterType !== "all") {
      filteredItems = items.filter(it => it.media_type === filterType);
    }

    // Module Déduplication & Tri séquentiel
    const dedupParam = req.query.dedup as string | undefined;
    const shouldDedup = dedupParam !== "false";
    let deduplicatedEpisodes: DeduplicatedEpisode[] = [];

    if (shouldDedup && filteredItems.length > 0) {
      deduplicatedEpisodes = await aiDeduplicateAndSortEpisodes(filteredItems, q || metadata.title);
    }

    // Regroupement par saison pour expérience MovieBox
    const seasonsMap: { [season: number]: DeduplicatedEpisode[] } = {};
    for (const ep of deduplicatedEpisodes) {
      const s = ep.season_number || 1;
      if (!seasonsMap[s]) seasonsMap[s] = [];
      seasonsMap[s].push(ep);
    }
    const seasonsList = Object.entries(seasonsMap).map(([seasonStr, eps]) => ({
      season: parseInt(seasonStr, 10),
      name: `Saison ${seasonStr}`,
      episodes_count: eps.length,
      episodes: eps,
    }));

    const requestedPageSize = parseInt(req.query.page_size as string || req.query.limit as string, 10);
    const pageSize = (!isNaN(requestedPageSize) && requestedPageSize > 0)
      ? Math.min(requestedPageSize, 500)
      : RESULTS_PAGE_SIZE;

    const totalResults = filteredItems.length;
    const totalPages = Math.max(1, Math.ceil(totalResults / pageSize));
    const finalPage = Math.min(pageNum, totalPages);
    const start = (finalPage - 1) * pageSize;
    const pageItems = filteredItems.slice(start, start + pageSize);
    const duplicatesEliminated = Math.max(0, filteredItems.length - deduplicatedEpisodes.length);
    const nextOffsetId = (req as any)._nextOffsetId || null;

    return res.json({
      query: q,
      anime_info: metadata,
      results: pageItems,
      episodes: pageItems, // Compatibilité ascendante NLSbox (ordonné Episode 1, 2, 3...)
      deduplicated_episodes: deduplicatedEpisodes,
      seasons: seasonsMap,
      seasons_list: seasonsList,
      duplicates_eliminated: duplicatesEliminated,
      ai_processed: isGeminiAvailable(),
      episodes_found: totalResults,
      total_found: totalResults,
      page: finalPage,
      page_size: pageSize,
      total_pages: totalPages,
      has_next: finalPage < totalPages,
      has_prev: finalPage > 1,
      next_offset_id: nextOffsetId,
      has_more_telegram: nextOffsetId !== null,
    });
  } catch (err: any) {
    console.error("Search error:", err);
    return res.status(500).json({ detail: err.message || "Internal server error" });
  }
});

// 3. Download / Stream endpoint with Telegram MTProto streaming + Range Header (206) support
app.get(["/download/:channel_id/:message_id", "/download/:channel_id/:message_id/:filename"], async (req: Request, res: Response) => {
  const { channel_id, message_id } = req.params;
  const msgIdNum = parseInt(message_id, 10);
  const isDownload = req.query.dl === "1";

  const client = await getTelegramClient();

  // Si canal et message Telegram réels
  if (client && channel_id && channel_id !== "-1001234567890" && !isNaN(msgIdNum)) {
    try {
      const entity = await client.getEntity(channel_id);
      const [msg] = await client.getMessages(entity, { ids: [msgIdNum] });

      const doc = msg?.media && ("document" in msg.media ? (msg.media as any).document : "video" in msg.media ? (msg.media as any).video : null);

      if (!msg || !doc) {
        return res.status(404).send("Média ou message Telegram introuvable dans ce canal.");
      }

      const totalSize = Number(doc.size || 0);
      const audioAttr = doc.attributes?.find((a: any) => a.className === "DocumentAttributeAudio");
      const videoAttr = doc.attributes?.find((a: any) => a.className === "DocumentAttributeVideo");
      const fileNameAttr = doc.attributes?.find((a: any) => a.fileName);

      let fileName = fileNameAttr?.fileName;
      if (!fileName && audioAttr && audioAttr.title) {
        fileName = `${audioAttr.performer ? audioAttr.performer + " - " : ""}${audioAttr.title}.mp3`;
      } else if (!fileName && msg.message) {
        const firstLine = msg.message.split("\n")[0].trim();
        if (firstLine.length > 3) {
          fileName = `${firstLine}.${audioAttr ? "mp3" : "mp4"}`;
        }
      }
      if (!fileName) {
        fileName = audioAttr ? `audio_${message_id}.mp3` : `video_${message_id}.mp4`;
      }

      // Nettoyer strictement les caractères interdits pour les systèmes d'exploitation (Windows, Mac, Linux, Android)
      // Caractères interdits : \ / : * ? " < > |
      const sanitizedName = fileName
        .replace(/[/\\?%*:|"<>]/g, "_")
        .replace(/\s+/g, " ")
        .trim();
      const asciiName = sanitizedName.replace(/[^\x20-\x7E]/g, "_").slice(0, 120);
      const encodedName = encodeURIComponent(sanitizedName);

      // Détecter ou affiner le type MIME
      let mimeType = doc.mimeType;
      const lowerName = sanitizedName.toLowerCase();
      if (!mimeType || mimeType === "application/octet-stream") {
        if (lowerName.endsWith(".mp4")) mimeType = "video/mp4";
        else if (lowerName.endsWith(".mp3")) mimeType = "audio/mpeg";
        else if (lowerName.endsWith(".mkv")) mimeType = "video/x-matroska";
        else if (lowerName.endsWith(".webm")) mimeType = "video/webm";
        else if (lowerName.endsWith(".flac")) mimeType = "audio/flac";
        else if (lowerName.endsWith(".pdf")) mimeType = "application/pdf";
        else if (audioAttr) mimeType = "audio/mpeg";
        else if (videoAttr) mimeType = "video/mp4";
        else mimeType = "application/octet-stream";
      }

      const disposition = isDownload
        ? `attachment; filename="${asciiName}"; filename*=UTF-8''${encodedName}`
        : `inline; filename="${asciiName}"; filename*=UTF-8''${encodedName}`;

      res.setHeader("Content-Disposition", disposition);
      res.setHeader("Accept-Ranges", "bytes");
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, Content-Disposition");
      res.setHeader("X-Content-Type-Options", "nosniff");

      const rangeHeader = req.headers.range;
      const requestSize = 512 * 1024; // 512KB Telegram chunk optimal

      if (rangeHeader) {
        let start = 0;
        let end = totalSize - 1;

        const parts = rangeHeader.replace(/bytes=/, "").trim().split("-");
        const partStart = parts[0];
        const partEnd = parts[1];

        if (partStart === "" && partEnd !== "") {
          // Suffix byte range: e.g. bytes=-500000 (derniers 500KB pour lire le box MOOV en fin de fichier MP4)
          const suffixLength = parseInt(partEnd, 10);
          start = Math.max(0, totalSize - suffixLength);
          end = totalSize - 1;
        } else {
          start = partStart ? parseInt(partStart, 10) : 0;
          end = partEnd ? parseInt(partEnd, 10) : totalSize - 1;
        }

        if (isNaN(start) || isNaN(end) || start >= totalSize || start > end) {
          res.setHeader("Content-Range", `bytes */${totalSize}`);
          return res.status(416).send("Range Not Satisfiable");
        }

        const chunkLength = end - start + 1;
        res.status(206);
        res.setHeader("Content-Range", `bytes ${start}-${end}/${totalSize}`);
        res.setHeader("Content-Length", chunkLength.toString());
        res.setHeader("Content-Type", mimeType);

        if (req.method === "HEAD") {
          return res.end();
        }

        const numChunks = Math.ceil(chunkLength / requestSize) + 1;

        const iter = client.iterDownload({
          file: msg.media,
          offset: bigInt(start),
          limit: numChunks,
          requestSize: requestSize,
        });

        let bytesSent = 0;
        let closed = false;
        req.on("close", () => {
          closed = true;
        });

        for await (const chunk of iter) {
          if (closed) break;
          const remaining = chunkLength - bytesSent;
          if (remaining <= 0) break;
          const toWrite = chunk.length > remaining ? chunk.subarray(0, remaining) : chunk;
          const canContinue = res.write(toWrite);
          bytesSent += toWrite.length;
          if (bytesSent >= chunkLength) break;
          if (!canContinue && !closed) {
            await new Promise((resolve) => res.once("drain", resolve));
          }
        }
        return res.end();
      } else {
        // Direct download complet ou streaming sans en-tête Range
        res.status(200);
        res.setHeader("Content-Length", totalSize.toString());
        res.setHeader("Content-Type", mimeType);

        if (req.method === "HEAD") {
          return res.end();
        }

        const numChunks = Math.ceil(totalSize / requestSize) + 1;

        const iter = client.iterDownload({
          file: msg.media,
          offset: bigInt(0),
          limit: numChunks,
          requestSize: requestSize,
        });

        let bytesSent = 0;
        let closed = false;
        req.on("close", () => {
          closed = true;
        });

        for await (const chunk of iter) {
          if (closed) break;
          const remaining = totalSize - bytesSent;
          if (remaining <= 0) break;
          const toWrite = chunk.length > remaining ? chunk.subarray(0, remaining) : chunk;
          const canContinue = res.write(toWrite);
          bytesSent += toWrite.length;
          if (bytesSent >= totalSize) break;
          if (!canContinue && !closed) {
            await new Promise((resolve) => res.once("drain", resolve));
          }
        }
        return res.end();
      }
    } catch (err: any) {
      console.error(`Download/Stream error for channel ${channel_id} msg ${message_id}:`, err?.message || err);
      if (!res.headersSent) {
        return res.status(500).send(`Erreur lors du streaming ou téléchargement Telegram: ${err.message || err}`);
      }
      return res.end();
    }
  }

  // Fallback vidéo de test si démonstration
  const totalSize = sampleVideoData.length;
  const rangeHeader = req.headers.range;
  let start = 0;
  let end = totalSize - 1;
  let statusCode = 200;

  if (rangeHeader) {
    const match = /bytes=(\d*)-(\d*)/.exec(rangeHeader);
    if (match) {
      if (match[1]) start = parseInt(match[1], 10);
      if (match[2]) end = parseInt(match[2], 10);
      statusCode = 206;
      res.setHeader("Content-Range", `bytes ${start}-${end}/${totalSize}`);
    }
  }

  const chunkLength = end - start + 1;
  res.status(statusCode);
  res.setHeader("Content-Type", "video/mp4");
  res.setHeader("Accept-Ranges", "bytes");
  res.setHeader("Content-Length", chunkLength.toString());
  res.send(sampleVideoData.subarray(start, end + 1));
});

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`NLSbox Pro Engine running at http://0.0.0.0:${PORT} (env: ${process.env.NODE_ENV || "development"})`);
});

// Arrêt propre (Graceful Shutdown) pour les déploiements Render
process.on("SIGTERM", async () => {
  console.log("SIGTERM reçu : fermeture progressive du serveur HTTP et de la session MTProto...");
  await closeTelegramClient();
  server.close(() => {
    console.log("Serveur HTTP fermé proprement.");
    process.exit(0);
  });
});

process.on("SIGINT", async () => {
  console.log("SIGINT reçu : fermeture du serveur HTTP et de la session MTProto...");
  await closeTelegramClient();
  server.close(() => {
    process.exit(0);
  });
});
