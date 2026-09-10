/**
 * NLSbox Sanitizer & Parser Engine
 * Nettoie les noms de fichiers pollués par les canaux Telegram,
 * extrait les métadonnées (saison, épisode, qualité, langue, team),
 * filtre le spam et ordonne chronologiquement (Épisode 1 en haut).
 */

export interface ParsedMedia {
  raw_name: string;
  clean_title: string;
  media_type: "video" | "audio" | "document" | "archive" | "file";
  series_name?: string;
  season_number: number | null;
  episode_number: number | null;
  artist?: string | null;
  track_title?: string | null;
  quality: string | null;     // '4K', '1080p', '720p', '480p', 'FLAC', '320kbps'
  language: string | null;    // 'VOSTFR', 'VF', 'MULTI', 'ENG'
  codec: string | null;       // 'x264', 'x265', 'HEVC', 'MP3', 'AAC', 'FLAC'
  release_group: string | null;
  is_movie: boolean;
  is_ova: boolean;
  relevance_score: number;
}

// Regex pour enlever les liens et mentions Telegram / Web
const TELEGRAM_LINKS_REGEX = /(?:https?:\/\/)?(?:www\.)?(?:t(?:elegram)?\.(?:me|dog)[/_][a-zA-Z0-9_+/.-]+)|\bt\.me[_\w.]+/gi;
const WEB_URL_REGEX = /https?:\/\/[^\s]+/gi;
const CHANNEL_MENTION_REGEX = /@([a-zA-Z0-9]+)[_\s-]+|@[a-zA-Z0-9_]{3,}\b/g;

// Regex pour enlever les extensions de fichiers
const EXTENSION_REGEX = /\.(mp4|mkv|avi|webm|ts|m4v|flv|mov|wmv|pdf|epub|cbr|cbz|mobi|zip|rar|tar|7z|mp3|flac|wav|m4a|aac|ogg|opus)$/i;

// Regex pour les résolutions
const QUALITY_PATTERNS: [RegExp, string][] = [
  [/(?:2160p|4k|uhd)/i, "4K"],
  [/(?:1080p|fhd|full[\s._-]?hd)/i, "1080p"],
  [/(?:720p|hd)/i, "720p"],
  [/(?:480p|sd)/i, "480p"],
  [/(?:360p)/i, "360p"],
];

// Regex pour les langues / sous-titres
const LANGUAGE_PATTERNS: [RegExp, string][] = [
  [/(?:\b|_|\[|\()vostfr(?:\b|_|\]|\))/i, "VOSTFR"],
  [/(?:\b|_|\[|\()multi(?:\b|_|\]|\))/i, "MULTI"],
  [/(?:\b|_|\[|\()truefrench(?:\b|_|\]|\))/i, "TRUEFRENCH"],
  [/(?:\b|_|\[|\()vf(?:\b|_|\]|\))/i, "VF"],
  [/(?:\b|_|\[|\()vff(?:\b|_|\]|\))/i, "VFF"],
  [/(?:\b|_|\[|\()eng[\s._-]?sub(?:\b|_|\]|\))/i, "ENG SUB"],
  [/(?:\b|_|\[|\()raw(?:\b|_|\]|\))/i, "RAW"],
];

// Regex pour les codecs
const CODEC_PATTERNS: [RegExp, string][] = [
  [/(?:x265|h265|hevc)/i, "x265"],
  [/(?:x264|h264|avc)/i, "x264"],
  [/(?:10bit|10-bit|hi10p)/i, "10bit"],
  [/(?:aac|ac3|eac3|flac|dts)/i, "AAC"],
];

// Regex pour Saison, Tome & Volume
const SEASON_PATTERNS = [
  /s(?:eason|aison)?[\s._-]*(\d{1,2})/i,
  /(?:tome|volume|vol)[\s._-]*(\d{1,3})/i,
  /(?:^|[\s._-])t(\d{1,3})(?:[\s._-]|$)/i,
  /(?:^|[\s._-])(\d{1,2})(?:e|x)\d{1,4}/i,
];

const EPISODE_PATTERNS = [
  /s\d{1,2}[\s._-]*e(?:p(?:isode)?)?[\s._-]*(\d{1,4})/i,
  /(?<![a-zA-Z0-9])(?:episode|ep|épisode|ép)[\s._-]*(\d{1,4})(?:v\d+)?(?![a-zA-Z0-9])/i,
  /(?<![a-zA-Z0-9])(?:chapitre|chap|ch|scan)[\s._-]*(\d{1,4})(?:v\d+)?(?![a-zA-Z0-9])/i,
  /(?<![a-zA-Z0-9])e[\s._-]*(\d{1,4})(?:v\d+)?(?![a-zA-Z0-9])/i,
  /(?:saison|season)[\s._-]*\d{1,2}[\s._-]+(?:e|ep)?[\s._-]*(\d{1,4})/i,
  /(?:^|[\[\(\s._-])-\s*(\d{1,4})(?:v\d+)?(?:[\]\)\s._-]|$)/, // standalone " - 05 - "
  /(?:\[|\()(\d{1,4})(?:v\d+)?(?:\]|\))/, // "[05]" ou "(05)"
  /(?:^|[._\s-])c[\s._-]*(\d{1,4})(?:v\d+)?(?:[._\s-]|$)/i, // "c01" ou "c 01"
  /(?:^|[._\s-])(\d{1,4})(?:v\d+)?(?:\.mp4|\.mkv|\.avi|\.pdf|\.cbr|\.cbz|\.epub|$)/i, // "Naruto 05.mp4"
];

export function sanitizeFileName(
  rawFileName: string,
  caption: string = "",
  mimeType: string = "",
  audioAttr?: { title?: string; performer?: string } | null
): ParsedMedia {
  let text = (rawFileName || "").trim();

  // Déterminer le type de média
  let mediaType: "video" | "audio" | "document" | "archive" | "file" = "file";
  const lowerMime = (mimeType || "").toLowerCase();
  const lowerName = text.toLowerCase();

  if (lowerMime.startsWith("audio/") || audioAttr || /\.(mp3|flac|wav|m4a|aac|ogg|wma|opus)$/i.test(lowerName)) {
    mediaType = "audio";
  } else if (lowerMime.startsWith("video/") || /\.(mp4|mkv|avi|webm|ts|m4v|flv|mov|wmv)$/i.test(lowerName)) {
    mediaType = "video";
  } else if (lowerMime.includes("pdf") || /\.(pdf|epub|cbr|cbz|mobi|docx?|xlsx?)$/i.test(lowerName)) {
    mediaType = "document";
  } else if (/\.(zip|rar|7z|tar|gz|bz2|xz|apk)$/i.test(lowerName)) {
    mediaType = "archive";
  }

  // Traitement audio spécifique si tags ID3 Telegram présents
  if (mediaType === "audio" && audioAttr && (audioAttr.title || audioAttr.performer)) {
    const artist = (audioAttr.performer || "").trim();
    const track = (audioAttr.title || "").trim();
    const cleanTitle = artist && track ? `${artist} - ${track}` : (track || artist || text);
    return {
      raw_name: rawFileName || `${cleanTitle}.mp3`,
      clean_title: cleanTitle,
      media_type: "audio",
      artist: artist || null,
      track_title: track || null,
      season_number: null,
      episode_number: null,
      quality: null,
      language: null,
      codec: lowerName.includes("flac") ? "FLAC" : "MP3",
      release_group: null,
      is_movie: false,
      is_ova: false,
      relevance_score: 0,
    };
  }

  // Si le nom de fichier est trop court ou générique (ex: "video.mp4"), on utilise la première ligne de la légende
  if ((text.length < 5 || text.toLowerCase().startsWith("video.")) && caption) {
    const firstLine = caption.split("\n")[0].trim();
    if (firstLine.length > 3) {
      text = firstLine;
    }
  }

  // 1. Extraire la release group au tout début si entre crochets (ex: "[Erai-raws] ...")
  let releaseGroup: string | null = null;
  const initialGroupMatch = /^\[([a-zA-Z0-9\s._-]+)\]/.exec(text);
  if (initialGroupMatch) {
    const candidate = initialGroupMatch[1].trim();
    // Éviter de confondre avec un tag de résolution ou langue
    if (!/^(1080p|720p|480p|vostfr|vf|multi|x264|x265)$/i.test(candidate)) {
      releaseGroup = candidate;
    }
  }

  // 2. Extraire la qualité
  let quality: string | null = null;
  for (const [pattern, label] of QUALITY_PATTERNS) {
    if (pattern.test(text)) {
      quality = label;
      break;
    }
  }

  // 3. Extraire la langue
  let language: string | null = null;
  for (const [pattern, label] of LANGUAGE_PATTERNS) {
    if (pattern.test(text)) {
      language = label;
      break;
    }
  }

  // 4. Extraire le codec
  let codec: string | null = null;
  for (const [pattern, label] of CODEC_PATTERNS) {
    if (pattern.test(text)) {
      codec = label;
      break;
    }
  }

  // 5. Détecter si c'est un Film ou OAV
  const isMovie = /(?:\b|_|\[|\()(?:movie|film|le[\s._-]film)(?:\b|_|\]|\))/i.test(text);
  const isOva = /(?:\b|_|\[|\()(?:ova|oav|special|sp)(?:\b|_|\]|\))/i.test(text);

  // Pré-dépollution du texte pour extraction fiable
  let workingText = text.replace(EXTENSION_REGEX, "");
  workingText = workingText.replace(TELEGRAM_LINKS_REGEX, " ");
  workingText = workingText.replace(WEB_URL_REGEX, " ");
  workingText = workingText.replace(CHANNEL_MENTION_REGEX, " ");
  const normalizedWorkingText = workingText.replace(/[._]+/g, " ");

  // 6. Extraire la saison / tome / volume
  let seasonNumber: number | null = null;
  let textForEpisode = normalizedWorkingText;
  for (const pattern of SEASON_PATTERNS) {
    const m = pattern.exec(normalizedWorkingText) || pattern.exec(text);
    if (m && m[1]) {
      const num = parseInt(m[1], 10);
      if (!isNaN(num) && num > 0 && num < 100) {
        seasonNumber = num;
        textForEpisode = textForEpisode.replace(m[0], " ");
        break;
      }
    }
  }

  // 7. Extraire le numéro d'épisode / chapitre
  let episodeNumber: number | null = null;
  for (const pattern of EPISODE_PATTERNS) {
    const m = pattern.exec(textForEpisode);
    if (m && (m[1] || m[2])) {
      const val = m[1] || m[2];
      const num = parseInt(val, 10);
      // Filtre les faux positifs (ex: années 1080, 2023, 2024)
      if (!isNaN(num) && num >= 0 && num < 2000 && num !== 1080 && num !== 720 && num !== 480 && num !== 2160) {
        episodeNumber = num;
        break;
      }
    }
  }

  // 8. NETTOYAGE PROFOND DU TITRE (Dépollution)
  let clean = workingText;

  // Enlever extension
  clean = clean.replace(EXTENSION_REGEX, "");

  // Enlever les liens Telegram et URL
  clean = clean.replace(TELEGRAM_LINKS_REGEX, " ");
  clean = clean.replace(WEB_URL_REGEX, " ");

  // Enlever les mentions de chaînes Telegram (@channel)
  clean = clean.replace(CHANNEL_MENTION_REGEX, " ");

  // Enlever les tags techniques entre crochets ou parenthèses
  clean = clean.replace(/\[[^\]]*(?:1080p|720p|480p|x264|x265|hevc|vostfr|vf|multi|web-dl|webrip|bluray|bdrip|aac|flac)[^\]]*\]/gi, " ");
  clean = clean.replace(/\([^)]*(?:1080p|720p|480p|x264|x265|hevc|vostfr|vf|multi|web-dl|webrip|bluray|bdrip|aac|flac)[^)]*\)/gi, " ");

  // Enlever le release group initial [Team]
  if (initialGroupMatch) {
    clean = clean.replace(initialGroupMatch[0], " ");
  }

  // Enlever les hashtags Telegram (#anime #naruto)
  clean = clean.replace(/#[a-zA-Z0-9_]+/g, " ");

  // Enlever les mots clés techniques parasites isolés
  clean = clean.replace(/\b(?:WEBRip|WEB-DL|BluRay|BDRip|HDTV|AAC2\.0|AAC|AC3|x264|x265|HEVC|10bit|Hi10P|VOSTFR|TRUEFRENCH|MULTI|VF|VFF|1080p|720p|480p|2160p|4K|HD|FHD|SD)\b/gi, " ");

  // Enlever crochets/parenthèses vides [ ], ( ), { }
  clean = clean.replace(/\[\s*\]/g, " ");
  clean = clean.replace(/\(\s*\)/g, " ");
  clean = clean.replace(/\{\s*\}/g, " ");

  // Enlever emojis décoratifs fréquents dans les canaux Telegram
  clean = clean.replace(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, " ");

  // Remplacer les points, underscores et tirets multiples par des espaces
  clean = clean.replace(/[._]+/g, " ");
  clean = clean.replace(/\s*-\s*-\s*/g, " - ");
  clean = clean.replace(/\s+/g, " ").trim();

  // Extraire le nom de la série (avant le numéro d'épisode / saison)
  let seriesName = clean;
  // Détection manga / scan
  const isManga = mediaType === "document" || /\.(pdf|cbr|cbz|epub)$/i.test(rawFileName) || /(?:scan|manga|tome|chapitre)/i.test(rawFileName);

  if (isManga) {
    seriesName = seriesName
      .replace(/(?:tome|volume|vol)\s*\d{1,3}/i, "")
      .replace(/(?:chapitre|chap|ch|scan)\s*\d{1,4}/i, "")
      .replace(/\s*-\s*$/, "")
      .replace(/^\s*-\s*/, "")
      .trim();
  }

  if (episodeNumber !== null) {
    // Retirer le motif d'épisode du titre principal
    seriesName = seriesName
      .replace(new RegExp(`(?:s(?:eason|aison)?\\s*\\d{1,2}\\s*)?(?:e|ep|episode)?\\s*0*${episodeNumber}\\b`, "i"), "")
      .replace(/\s*-\s*$/, "")
      .replace(/^\s*-\s*/, "")
      .trim();
  }

  // Nettoyage final des crochets et tirets orphelins
  seriesName = seriesName
    .replace(/^[[\](){}\s-_]+|[[\](){}\s-_]+$/g, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!seriesName || seriesName.length < 2) {
    seriesName = (rawFileName || "Média").replace(/\.[a-zA-Z0-9]+$/, "").trim();
  }

  // Formater un titre propre et élégant pour l'utilisateur
  let cleanTitle = seriesName;
  if (isManga) {
    if (seasonNumber !== null) {
      const sPad = seasonNumber < 10 ? `0${seasonNumber}` : `${seasonNumber}`;
      cleanTitle += ` - Tome ${sPad}`;
    }
    if (episodeNumber !== null) {
      const epPad = episodeNumber < 10 ? `0${episodeNumber}` : `${episodeNumber}`;
      cleanTitle += ` - Chapitre ${epPad}`;
    }
  } else {
    if (seasonNumber && seasonNumber > 1) {
      cleanTitle += ` - Saison ${seasonNumber}`;
    }

    if (episodeNumber !== null) {
      const epPad = episodeNumber < 10 ? `0${episodeNumber}` : `${episodeNumber}`;
      cleanTitle += ` - Épisode ${epPad}`;
    } else if (isMovie) {
      cleanTitle += ` - Film`;
    } else if (isOva) {
      cleanTitle += ` - OAV`;
    }
  }

  return {
    raw_name: rawFileName || cleanTitle,
    clean_title: cleanTitle,
    media_type: mediaType,
    series_name: seriesName,
    season_number: seasonNumber,
    episode_number: episodeNumber,
    quality,
    language,
    codec,
    release_group: releaseGroup,
    is_movie: isMovie,
    is_ova: isOva,
    relevance_score: 0,
  };
}

/**
 * Filtre les faux-positifs et ordonne chronologiquement :
 * - Épisode 1 en premier (croissant : 1, 2, 3...)
 * - Filtre le spam (élimine les fichiers qui n'ont rien à voir avec la recherche)
 */
export function filterAndSortEpisodes<T extends { file_name: string; caption?: string; mime_type?: string; audio_attr?: { title?: string; performer?: string } | null }>(
  items: T[],
  searchQuery: string = ""
): (T & ParsedMedia)[] {
  const query = (searchQuery || "").toLowerCase().trim();
  const queryTokens = query
    .split(/\s+/)
    .filter(t => t.length > 1)
    .map(t => t.toLowerCase());

  const enriched = items.map(item => {
    const parsed = sanitizeFileName(item.file_name, item.caption, item.mime_type, item.audio_attr);

    // Calcul du score de pertinence anti-spam
    const fullSearchable = `${parsed.series_name || ""} ${parsed.clean_title} ${parsed.artist || ""} ${item.file_name} ${item.caption || ""}`.toLowerCase();
    
    let score = 0;
    let matchingTokens = 0;

    if (queryTokens.length === 0) {
      score = 100; // Pas de filtre particulier, tout est pertinent
    } else {
      for (const token of queryTokens) {
        if (fullSearchable.includes(token)) {
          matchingTokens++;
          score += 100;
        }
      }

      const titleLower = parsed.clean_title.toLowerCase();
      if (titleLower === query) {
        score += 1000;
      } else if (titleLower.startsWith(query)) {
        score += 500;
      } else if (titleLower.includes(query)) {
        score += 250;
      }
    }

    parsed.relevance_score = score;
    return { ...item, ...parsed };
  });

  // 1. Filtrage : Si des mots clés ont été fournis, garder les résultats pertinents
  const filtered = enriched.filter(item => {
    if (queryTokens.length === 0) return true;
    return item.relevance_score > 0;
  });

  // 2. Tri MovieBox séquentiel et ordonné (Épisode 1, 2, 3... Tome 1, 2, 3...)
  filtered.sort((a, b) => {
    // 1. Comparaison par saison / volume (Saison 1 en premier)
    const seasonA = a.season_number ?? 1;
    const seasonB = b.season_number ?? 1;
    if (a.season_number !== null || b.season_number !== null) {
      if (seasonA !== seasonB) return seasonA - seasonB;
    }

    // 2. Comparaison par épisode / chapitre (Épisode 1 en premier)
    if (a.episode_number !== null && b.episode_number !== null) {
      if (a.episode_number !== b.episode_number) {
        return a.episode_number - b.episode_number;
      }
    } else if (a.episode_number !== null) {
      return -1; // Les épisodes numérotés passent avant les bonus non identifiés
    } else if (b.episode_number !== null) {
      return 1;
    }

    // 3. Tri alphanumérique naturel (ex: "01" avant "02" avant "10")
    return (a.clean_title || a.file_name).localeCompare(
      b.clean_title || b.file_name,
      undefined,
      { numeric: true, sensitivity: "base" }
    );
  });

  return filtered;
}
