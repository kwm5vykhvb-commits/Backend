import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Client singleton avec User-Agent télémétrie requis
let aiClient: GoogleGenAI | null = null;

export function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey.trim() === "") {
    return null;
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: apiKey.trim(),
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

export function isGeminiAvailable(): boolean {
  return !!(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim().length > 5);
}

// Modèles supportés et gestion de la saturation (503 / High Demand)
const OVERLOAD_COOLDOWN_MS = 3 * 60 * 1000;
const overloadedModels: Record<string, number> = {
  // Prédéclarer gemini-3.8-flash en cooldown initial pour router directement sur flash-lite
  // tout en sondant à nouveau le modèle 3.8 dès que le pic de charge Google s'estompe.
  "gemini-3.8-flash": Date.now() + OVERLOAD_COOLDOWN_MS,
};

export function markModelOverloaded(model: string, cooldownMs: number = OVERLOAD_COOLDOWN_MS): void {
  overloadedModels[model] = Date.now() + cooldownMs;
}

export function isModelOverloaded(model: string): boolean {
  return (overloadedModels[model] || 0) > Date.now();
}

export function getActiveGeminiModel(): string {
  if (!isModelOverloaded("gemini-3.8-flash")) {
    return "gemini-3.8-flash";
  }
  return "gemini-3.1-flash-lite";
}

export function getCandidateModels(preferredModel: string = "gemini-3.8-flash"): string[] {
  const now = Date.now();
  const pool = [preferredModel, "gemini-3.1-flash-lite", "gemini-flash-latest"];
  const unique = Array.from(new Set(pool));
  return unique.sort((a, b) => {
    const aCool = (overloadedModels[a] || 0) > now ? 1 : 0;
    const bCool = (overloadedModels[b] || 0) > now ? 1 : 0;
    return aCool - bCool;
  });
}

/**
 * Exécute generateContent avec bascule automatique transparente sur modèle de secours
 * si le modèle principal est en 503 (haute demande / saturation temporaire).
 */
export async function generateContentWithFallback(
  ai: GoogleGenAI,
  params: {
    preferredModel?: string;
    contents: any;
    config?: any;
  }
) {
  const candidateModels = getCandidateModels(params.preferredModel || "gemini-3.8-flash");
  let lastError: any = null;

  for (const model of candidateModels) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: params.contents,
        config: params.config,
      });
      // Succès : si le modèle était en cooldown, lever la restriction
      if (overloadedModels[model]) {
        delete overloadedModels[model];
      }
      return response;
    } catch (err: any) {
      lastError = err;
      const errMsg = err?.message || JSON.stringify(err);
      const is503 =
        errMsg.includes("503") ||
        errMsg.includes("high demand") ||
        errMsg.includes("UNAVAILABLE") ||
        err?.status === "UNAVAILABLE" ||
        err?.code === 503;

      if (is503) {
        markModelOverloaded(model);
        console.log(`[Gemini AI] Modèle ${model} temporairement saturé (503), bascule vers le modèle suivant...`);
        continue;
      }
      console.warn(`[Gemini AI] Avertissement sur ${model}, test du modèle alternatif:`, errMsg);
    }
  }

  throw lastError;
}

/**
 * Exécute un échange de chat avec bascule transparente en cas de saturation de modèle
 */
export async function sendChatMessageWithFallback(
  ai: GoogleGenAI,
  systemInstruction: string,
  formattedHistory: any[],
  userMessage: string
): Promise<string> {
  const candidateModels = getCandidateModels("gemini-3.8-flash");
  let lastError: any = null;

  for (const model of candidateModels) {
    try {
      const chat = ai.chats.create({
        model,
        config: { systemInstruction },
        history: formattedHistory,
      });
      const response = await chat.sendMessage({ message: userMessage });
      if (overloadedModels[model]) {
        delete overloadedModels[model];
      }
      return response.text || "Je n'ai pas pu formuler de réponse. Veuillez réessayer.";
    } catch (err: any) {
      lastError = err;
      const errMsg = err?.message || JSON.stringify(err);
      if (errMsg.includes("503") || errMsg.includes("high demand") || err?.code === 503) {
        markModelOverloaded(model);
        console.log(`[Gemini Chat] Modèle ${model} temporairement saturé (503), bascule vers le modèle suivant...`);
        continue;
      }
    }
  }

  throw lastError;
}

export interface MediaItemInput {
  message_id: number;
  channel_id: string;
  file_name: string;
  caption?: string;
  size_mb?: number;
  mime_type?: string;
  duration?: number | null;
  raw_name?: string;
  clean_title?: string;
  series_name?: string;
  season_number?: number | null;
  episode_number?: number | null;
  quality?: string | null;
  language?: string | null;
  codec?: string | null;
  stream_url?: string;
  download_url?: string;
}

export interface DeduplicatedEpisode {
  canonical_id: string; // Ex: "naruto-s1-e1"
  series_name: string;
  season_number: number;
  episode_number: number;
  clean_title: string; // Format standardisé: "Naruto - S01E01"
  language: string | null;
  best_quality: string | null;
  primary_file: MediaItemInput;
  variants: {
    quality: string | null;
    language: string | null;
    size_mb?: number;
    message_id: number;
    stream_url?: string;
    download_url?: string;
    raw_name: string;
  }[];
  duplicates_merged: number;
  ai_processed: boolean;
}

export interface PosterIdentificationResult {
  detected_series: string;
  season?: number | null;
  episode?: number | null;
  confidence: number; // 0 - 100
  language?: string | null;
  synopsis?: string;
  genres?: string[];
  detected_from: "poster" | "thumbnail" | "caption_vision";
}

/**
 * 1. DÉDUPLICATION ET RENOMMAGE ORDONNÉ AVEC GEMINI
 * Regroupe les variantes identiques (ex: 'naruto ep1', 'Naruto S1 épisode 1', '[1080p] Naruto 01'),
 * génère un titre standardisé propre et ordonne chronologiquement du 1er au dernier épisode.
 */
export async function aiDeduplicateAndSortEpisodes(
  items: MediaItemInput[],
  seriesHint: string = ""
): Promise<DeduplicatedEpisode[]> {
  if (!items || items.length === 0) return [];

  const ai = getGeminiClient();

  // Si l'API Gemini n'est pas configurée, bascule vers l'algorithme heuristique local
  if (!ai) {
    return fallbackDeduplicateAndSort(items);
  }

  try {
    // Préparer un résumé compact pour Gemini afin de minimiser les tokens et la latence
    const candidates = items.map((it, idx) => ({
      index: idx,
      file_name: it.file_name,
      caption: (it.caption || "").slice(0, 150),
      size_mb: it.size_mb,
      season_hint: it.season_number,
      episode_hint: it.episode_number,
      quality_hint: it.quality,
      language_hint: it.language,
    }));

    const prompt = `Tu es le moteur d'organisation multimédia NLSbox.
Analyse cette liste de fichiers vidéo provenant de canaux Telegram d'anime/série.
Dans les canaux Telegram, un même épisode est souvent publié sous plusieurs formats différents avec des noms variés (ex: "naruto ep1.mp4", "Naruto S1 épisode 1 [1080p].mkv", "01.mp4").

Tâche :
1. Regroupe les fichiers qui correspondent au MÊME épisode de la même série (déduplication intelligente).
2. Pour chaque groupe d'épisode unique :
   - Détermine le nom exact de la série (anime). Utilise "${seriesHint}" comme référence si pertinent.
   - Détermine la saison (par défaut 1 si non spécifié).
   - Détermine le numéro d'épisode exact (entier > 0).
   - Normalise le titre standardisé au format : "{Nom de la série} - S{saison sur 2 chiffres}E{épisode sur 2 chiffres}" (ex: "Naruto - S01E01").
   - Identifie la langue (VOSTFR, VF, MULTI, ENG SUB, RAW) et la qualité (1080p, 720p, 480p, 4K).
   - Liste les index des fichiers appartenant à cet épisode, en désignant comme "primary_index" celui ayant la meilleure qualité ou taille.
3. Trie chronologiquement le tableau du premier épisode au dernier (S01E01, S01E02, S01E03...).

Liste des fichiers :
${JSON.stringify(candidates, null, 2)}`;

    const response = await generateContentWithFallback(ai, {
      preferredModel: "gemini-3.8-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              series_name: { type: Type.STRING },
              season_number: { type: Type.INTEGER },
              episode_number: { type: Type.INTEGER },
              clean_title: { type: Type.STRING },
              language: { type: Type.STRING },
              best_quality: { type: Type.STRING },
              primary_index: { type: Type.INTEGER },
              matched_indices: {
                type: Type.ARRAY,
                items: { type: Type.INTEGER },
              },
            },
            required: [
              "series_name",
              "season_number",
              "episode_number",
              "clean_title",
              "primary_index",
              "matched_indices",
            ],
          },
        },
      },
    });

    const parsedJson = JSON.parse(response.text || "[]");

    if (!Array.isArray(parsedJson) || parsedJson.length === 0) {
      return fallbackDeduplicateAndSort(items);
    }

    const result: DeduplicatedEpisode[] = [];
    const usedIndices = new Set<number>();

    for (const group of parsedJson) {
      const primaryIdx = group.primary_index ?? group.matched_indices?.[0] ?? 0;
      const primaryItem = items[primaryIdx] || items[0];

      const variants = (group.matched_indices || [primaryIdx])
        .map((idx: number) => {
          usedIndices.add(idx);
          const it = items[idx];
          if (!it) return null;
          return {
            quality: it.quality || group.best_quality || null,
            language: it.language || group.language || null,
            size_mb: it.size_mb,
            message_id: it.message_id,
            stream_url: it.stream_url,
            download_url: it.download_url,
            raw_name: it.file_name,
          };
        })
        .filter(Boolean) as DeduplicatedEpisode["variants"];

      const seasonNum = Math.max(1, group.season_number || 1);
      const epNum = Math.max(1, group.episode_number || 1);
      const sPad = seasonNum < 10 ? `0${seasonNum}` : `${seasonNum}`;
      const ePad = epNum < 10 ? `0${epNum}` : `${epNum}`;
      const cleanTitle = group.clean_title || `${group.series_name || "Anime"} - S${sPad}E${ePad}`;

      result.push({
        canonical_id: `${(group.series_name || "series").toLowerCase().replace(/[^a-z0-9]/g, "_")}_s${seasonNum}_e${epNum}`,
        series_name: group.series_name || seriesHint || "Anime",
        season_number: seasonNum,
        episode_number: epNum,
        clean_title: cleanTitle,
        language: group.language || primaryItem.language || "VOSTFR",
        best_quality: group.best_quality || primaryItem.quality || "1080p",
        primary_file: {
          ...primaryItem,
          clean_title: cleanTitle,
          series_name: group.series_name || primaryItem.series_name,
          season_number: seasonNum,
          episode_number: epNum,
        },
        variants,
        duplicates_merged: Math.max(0, variants.length - 1),
        ai_processed: true,
      });
    }

    // Réintégrer les éléments restants qui n'auraient pas été matchés
    for (let i = 0; i < items.length; i++) {
      if (!usedIndices.has(i)) {
        const item = items[i];
        const sNum = item.season_number || 1;
        const eNum = item.episode_number || i + 1;
        result.push({
          canonical_id: `item_${item.message_id}_s${sNum}_e${eNum}`,
          series_name: item.series_name || seriesHint || "Contenu",
          season_number: sNum,
          episode_number: eNum,
          clean_title: item.clean_title || item.file_name,
          language: item.language || null,
          best_quality: item.quality || null,
          primary_file: item,
          variants: [
            {
              quality: item.quality || null,
              language: item.language || null,
              size_mb: item.size_mb,
              message_id: item.message_id,
              stream_url: item.stream_url,
              download_url: item.download_url,
              raw_name: item.file_name,
            },
          ],
          duplicates_merged: 0,
          ai_processed: false,
        });
      }
    }

    // Tri chronologique strict : Saison 1 -> Épisode 1, 2, 3...
    result.sort((a, b) => {
      if (a.season_number !== b.season_number) {
        return a.season_number - b.season_number;
      }
      return a.episode_number - b.episode_number;
    });

    return result;
  } catch (err: any) {
    console.warn("[Gemini AI] Échec du tri dédupliqué, bascule sur le tri local:", err?.message || err);
    return fallbackDeduplicateAndSort(items);
  }
}

/**
 * 2. RECONNAISSANCE VISUELLE D'UN ANIME VIA SON AFFICHE / VIGNETTE TELEGRAM (Gemini Multimodal Vision)
 * Quand un fichier est nommé sans le nom de l'anime (ex: "01.mp4", "episode_05.mkv"),
 * Gemini analyse visuellement l'affiche/poster pour identifier la série.
 */
export async function aiIdentifyAnimeFromPoster(
  imageBase64: string,
  mimeType: string = "image/jpeg",
  filenameHint: string = "",
  captionHint: string = ""
): Promise<PosterIdentificationResult | null> {
  const ai = getGeminiClient();
  if (!ai || !imageBase64) return null;

  try {
    const prompt = `Tu es un expert en animes et médias visuels japonais/internationaux.
Analyse cette image qui est une affiche, jaquette ou vignette d'un canal Telegram.
Fichier associé : "${filenameHint}"
Légende associée : "${captionHint}"

Identifie avec certitude :
1. Le nom officiel de l'anime/série (en alphabet latin / titre international populaire, ex: "Jujutsu Kaisen", "Demon Slayer", "One Piece", "Solo Leveling").
2. La saison (si identifiable sur l'affiche, ex: "Saison 2" ou "Cour 2").
3. Le numéro d'épisode si mentionné sur l'image ou la vignette.
4. Les genres principaux (Action, Shonen, etc.).
5. Un court synopsis en français (1 ou 2 phrases).
6. Un score de certitude entre 0 et 100.`;

    const response = await generateContentWithFallback(ai, {
      preferredModel: "gemini-3.8-flash",
      contents: [
        {
          inlineData: {
            mimeType: mimeType.startsWith("image/") ? mimeType : "image/jpeg",
            data: imageBase64,
          },
        },
        { text: prompt },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            detected_series: { type: Type.STRING },
            season: { type: Type.INTEGER },
            episode: { type: Type.INTEGER },
            confidence: { type: Type.INTEGER },
            genres: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            synopsis: { type: Type.STRING },
          },
          required: ["detected_series", "confidence"],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    if (!parsed.detected_series) return null;

    return {
      detected_series: parsed.detected_series,
      season: parsed.season ?? 1,
      episode: parsed.episode ?? null,
      confidence: parsed.confidence ?? 85,
      synopsis: parsed.synopsis || "",
      genres: parsed.genres || ["Anime", "Action"],
      detected_from: "poster",
    };
  } catch (err: any) {
    console.warn("[Gemini Vision] Échec de l'identification de l'affiche:", err?.message || err);
    return null;
  }
}

/**
 * Algorithme de déduplication heuristique local (sans API) utilisé si Gemini n'est pas joignable.
 */
function fallbackDeduplicateAndSort(items: MediaItemInput[]): DeduplicatedEpisode[] {
  const groups = new Map<string, MediaItemInput[]>();

  for (const item of items) {
    const s = item.season_number || 1;
    const ep = item.episode_number !== null ? item.episode_number : null;

    // Clé de groupe : nom de la série normalisé + saison + épisode
    const baseSeries = (item.series_name || "media")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");

    const groupKey = ep !== null ? `${baseSeries}_s${s}_e${ep}` : `item_${item.message_id}`;

    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey)!.push(item);
  }

  const result: DeduplicatedEpisode[] = [];

  for (const [key, groupItems] of groups.entries()) {
    // Trier les variantes : privilégier 1080p > 720p > 480p
    groupItems.sort((a, b) => {
      const qScore = (q: string | null | undefined) => {
        if (!q) return 0;
        if (q.includes("4K") || q.includes("2160")) return 4;
        if (q.includes("1080")) return 3;
        if (q.includes("720")) return 2;
        if (q.includes("480")) return 1;
        return 0;
      };
      const scoreDiff = qScore(b.quality) - qScore(a.quality);
      if (scoreDiff !== 0) return scoreDiff;
      return (b.size_mb || 0) - (a.size_mb || 0);
    });

    const primary = groupItems[0];
    const sNum = primary.season_number || 1;
    const epNum = primary.episode_number || 1;
    const sPad = sNum < 10 ? `0${sNum}` : `${sNum}`;
    const epPad = epNum < 10 ? `0${epNum}` : `${epNum}`;

    const cleanTitle = primary.clean_title || `${primary.series_name || "Anime"} - S${sPad}E${epPad}`;

    result.push({
      canonical_id: key,
      series_name: primary.series_name || "Anime",
      season_number: sNum,
      episode_number: epNum,
      clean_title: cleanTitle,
      language: primary.language || "VOSTFR",
      best_quality: primary.quality || "1080p",
      primary_file: primary,
      variants: groupItems.map((it) => ({
        quality: it.quality || null,
        language: it.language || null,
        size_mb: it.size_mb,
        message_id: it.message_id,
        stream_url: it.stream_url,
        download_url: it.download_url,
        raw_name: it.file_name,
      })),
      duplicates_merged: Math.max(0, groupItems.length - 1),
      ai_processed: false,
    });
  }

  // Tri chronologique croissant
  result.sort((a, b) => {
    if (a.season_number !== b.season_number) {
      return a.season_number - b.season_number;
    }
    return a.episode_number - b.episode_number;
  });

  return result;
}

export interface SmartSearchResult {
  interpreted_intent: string;
  target_series: string;
  season?: number | null;
  episode?: number | null;
  language?: string | null;
  quality?: string | null;
  search_keywords: string[];
  recommendations: {
    title: string;
    reason: string;
    genres: string[];
  }[];
}

/**
 * 3. RECHERCHE EN LANGAGE NATUREL & RECOMMANDATIONS (Smart AI Search)
 * Comprend les requêtes floues de l'utilisateur (ex: "anime d'exorcisme en VF", "épisode où L meurt", "série de ninjas")
 * et les traduit en critères de recherche précis avec des recommandations d'animes.
 */
export async function aiSmartSearchAssistant(
  userQuery: string,
  availableMediaTitles: string[] = []
): Promise<SmartSearchResult> {
  const cleanQ = (userQuery || "").trim();
  if (!cleanQ) {
    return {
      interpreted_intent: "Recherche générale",
      target_series: "",
      search_keywords: [],
      recommendations: [],
    };
  }

  const ai = getGeminiClient();
  if (!ai) {
    return {
      interpreted_intent: `Recherche locale pour "${cleanQ}"`,
      target_series: cleanQ,
      search_keywords: cleanQ.split(/\s+/).filter((w) => w.length > 2),
      recommendations: [
        { title: "Death Note", reason: "Classique incontournable du thriller psychologique", genres: ["Thriller", "Surnaturel"] },
        { title: "Jujutsu Kaisen", reason: "Anime d'action et d'exorcisme moderne", genres: ["Action", "Shonen"] },
        { title: "Solo Leveling", reason: "Dark fantasy et montée en puissance spectaculaire", genres: ["Action", "Fantasy"] },
      ],
    };
  }

  try {
    const prompt = `Tu es l'assistant de recherche intelligent du lecteur multimédia NLSbox.
L'utilisateur formule une recherche en langage naturel : "${cleanQ}".

Contenus disponibles en référence : ${JSON.stringify(availableMediaTitles.slice(0, 30))}

Tâches :
1. Déduis précisément l'intention de l'utilisateur (ex: série spécifique, saison, épisode précis, genre, langue comme VF ou VOSTFR).
2. Fournis le nom standardisé de la série principale recherchée.
3. Extrais les mots-clés optimaux pour filtrer des fichiers Telegram.
4. Si pertinent, suggère 3 animes recommandés correspondant aux goûts ou thèmes mentionnés avec la raison.`;

    const response = await generateContentWithFallback(ai, {
      preferredModel: "gemini-3.8-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            interpreted_intent: { type: Type.STRING },
            target_series: { type: Type.STRING },
            season: { type: Type.INTEGER },
            episode: { type: Type.INTEGER },
            language: { type: Type.STRING },
            quality: { type: Type.STRING },
            search_keywords: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            recommendations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  reason: { type: Type.STRING },
                  genres: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: ["title", "reason"],
              },
            },
          },
          required: ["interpreted_intent", "target_series", "search_keywords"],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    return {
      interpreted_intent: parsed.interpreted_intent || `Recherche pour "${cleanQ}"`,
      target_series: parsed.target_series || cleanQ,
      season: parsed.season ?? null,
      episode: parsed.episode ?? null,
      language: parsed.language ?? null,
      quality: parsed.quality ?? null,
      search_keywords: parsed.search_keywords || [cleanQ],
      recommendations: parsed.recommendations || [],
    };
  } catch (err: any) {
    console.warn("[Gemini AI] Erreur lors de la recherche intelligente:", err?.message || err);
    return {
      interpreted_intent: `Recherche locale pour "${cleanQ}"`,
      target_series: cleanQ,
      search_keywords: cleanQ.split(/\s+/).filter((w) => w.length > 2),
      recommendations: [],
    };
  }
}

export interface ChatMessage {
  role: "user" | "model";
  content: string;
}

/**
 * 4. ASSISTANT CONVERSATIONNEL IA NLSBOX (Chatbot Spécialisé Animes & Médias)
 * Répond aux questions sur les ordres de visionnage (canon vs filler), résumés d'arcs,
 * recommandations personnalisées et orientation dans les épisodes.
 */
export async function aiChatAssistant(
  userMessage: string,
  history: ChatMessage[] = [],
  contextMediaTitles: string[] = []
): Promise<string> {
  const ai = getGeminiClient();
  if (!ai) {
    return "Le module IA Gemini nécessite la variable d'environnement `GEMINI_API_KEY` pour converser. Vous pouvez définir votre clé API dans les paramètres du projet.";
  }

  try {
    const formattedHistory = history.map((m) => ({
      role: m.role,
      parts: [{ text: m.content }],
    }));

    const systemInstruction = `Tu es l'assistant IA officiel de NLSbox, la plateforme de streaming et d'exploration d'animes et médias connectée à Telegram.
Ton rôle :
- Aider l'utilisateur à trouver ses animes, séries, films ou musiques préférés.
- Expliquer les ordres de visionnage chronologiques (ex: saisons, OAV, films, épisodes canons vs fillers).
- Fournir des résumés captivants sans spoiler les moments clés (sauf si demandé explicitement).
- Recommander des pépites basées sur les préférences de l'utilisateur.
- Répondre avec un ton chaleureux, enthousiaste, expert en animes et en culture pop, avec un formatage Markdown soigné (listes à puces, gras).
- Contenus actuellement disponibles sur la chaîne : ${contextMediaTitles.slice(0, 20).join(", ") || "Catalogue universel NLSbox"}.`;

    return await sendChatMessageWithFallback(
      ai,
      systemInstruction,
      formattedHistory,
      userMessage
    );
  } catch (err: any) {
    console.error("[Gemini Chat] Erreur de conversation:", err?.message || err);
    return `Erreur lors de la communication avec Gemini : ${err?.message || "Erreur interne"}`;
  }
}

export interface EnrichedMetadataResult {
  canonical_title: string;
  original_title: string;
  synopsis: string;
  genres: string[];
  release_year: number;
  studio: string;
  rating: string;
  total_episodes: number;
  watch_order_tips: string;
}

/**
 * 5. ENRICHISSEMENT DE MÉTADONNÉES IA (Synopsis, Genres, Studio, Ordre de visionnage)
 */
export async function aiEnrichAnimeMetadata(
  seriesName: string
): Promise<EnrichedMetadataResult | null> {
  const ai = getGeminiClient();
  if (!ai || !seriesName.trim()) return null;

  try {
    const prompt = `Tu es une encyclopédie vivante de l'animation japonaise et internationale.
Génère la fiche détaillée et officielle pour l'anime/série suivante : "${seriesName}".
Donne un synopsis captivant en français, les genres, l'année, le studio d'animation, la note moyenne approximative, le nombre officiel d'épisodes et un conseil pour l'ordre de visionnage.`;

    const response = await generateContentWithFallback(ai, {
      preferredModel: "gemini-3.8-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            canonical_title: { type: Type.STRING },
            original_title: { type: Type.STRING },
            synopsis: { type: Type.STRING },
            genres: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            release_year: { type: Type.INTEGER },
            studio: { type: Type.STRING },
            rating: { type: Type.STRING },
            total_episodes: { type: Type.INTEGER },
            watch_order_tips: { type: Type.STRING },
          },
          required: ["canonical_title", "synopsis", "genres", "release_year"],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    return {
      canonical_title: parsed.canonical_title || seriesName,
      original_title: parsed.original_title || "",
      synopsis: parsed.synopsis || "Aucun synopsis disponible.",
      genres: parsed.genres || ["Anime"],
      release_year: parsed.release_year || new Date().getFullYear(),
      studio: parsed.studio || "Inconnu",
      rating: parsed.rating || "8.5/10",
      total_episodes: parsed.total_episodes || 24,
      watch_order_tips: parsed.watch_order_tips || "Suivre l'ordre chronologique des saisons.",
    };
  } catch (err: any) {
    console.warn("[Gemini AI] Erreur enrichissement:", err?.message || err);
    return null;
  }
}

