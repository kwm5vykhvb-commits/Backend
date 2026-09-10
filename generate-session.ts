/**
 * NLSbox - Générateur de SESSION_STRING Telegram MTProto (GramJS)
 * 
 * Permet de générer une nouvelle session Telegram propre pour remplacer
 * une session expirée ou invalidée (erreur 406: AUTH_KEY_DUPLICATED).
 * 
 * Usage: npm run session
 */

import { TelegramClient } from "telegram";
import { StringSession } from "telegram/sessions/index.js";
import readline from "readline";
import dotenv from "dotenv";

dotenv.config();

function askQuestion(query: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(query, (ans) => {
      rl.close();
      resolve(ans.trim());
    });
  });
}

async function main() {
  console.log("\n=======================================================");
  console.log("   🚀 NLSbox - Assistant de Session Telegram MTProto   ");
  console.log("=======================================================\n");
  console.log("Cet outil permet de générer une nouvelle SESSION_STRING propre.");
  console.log("Cela résout définitivement l'erreur 406: AUTH_KEY_DUPLICATED.\n");

  const envApiId = process.env.API_ID || "";
  const envApiHash = process.env.API_HASH || "";

  let apiIdStr = await askQuestion(`API_ID [défaut: ${envApiId || "à saisir"}]: `);
  apiIdStr = apiIdStr || envApiId;
  const apiId = parseInt(apiIdStr, 10);

  if (!apiId || isNaN(apiId)) {
    console.error("❌ API_ID invalide.");
    process.exit(1);
  }

  let apiHash = await askQuestion(`API_HASH [défaut: ${envApiHash ? envApiHash.slice(0, 6) + "..." : "à saisir"}]: `);
  apiHash = apiHash || envApiHash;

  if (!apiHash) {
    console.error("❌ API_HASH invalide.");
    process.exit(1);
  }

  const stringSession = new StringSession("");
  const client = new TelegramClient(stringSession, apiId, apiHash, {
    connectionRetries: 3,
    useWSS: false,
  });

  console.log("\n📡 Connexion aux serveurs Telegram...");
  await client.connect();

  console.log("✅ Connecté aux serveurs Telegram.");
  console.log("Veuillez saisir votre numéro de téléphone (au format international, ex: +33612345678) :");

  await client.start({
    phoneNumber: async () => await askQuestion("Numéro de téléphone (+...) : "),
    password: async () => await askQuestion("Mot de passe 2FA (si activé, sinon Entrée) : "),
    phoneCode: async () => await askQuestion("Code de vérification Telegram reçu : "),
    onError: (err) => console.error("Erreur Telegram:", err.message),
  });

  console.log("\n🎉 Connexion réussie !");
  const me = await client.getMe();
  console.log(`👤 Connecté en tant que : ${(me as any).firstName || ""} (@${(me as any).username || (me as any).id})`);

  const newSessionString = client.session.save() as unknown as string;

  console.log("\n=======================================================");
  console.log("🔑 VOTRE NOUVELLE SESSION_STRING :");
  console.log("=======================================================\n");
  console.log(newSessionString);
  console.log("\n=======================================================");
  console.log("👉 ÉTAPES SUIVANTES :");
  console.log("1. Copiez la clé ci-dessus.");
  console.log("2. Sur Render (dashboard.render.com) > Web Service > Environment :");
  console.log("   Mettez à jour la variable SESSION_STRING avec cette nouvelle valeur.");
  console.log("3. Dans l'application NLSbox, vous pouvez aussi la coller directement");
  console.log("   depuis l'interface pour tester immédiatement.");
  console.log("=======================================================\n");

  await client.disconnect();
  await client.destroy();
  process.exit(0);
}

main().catch((err) => {
  console.error("❌ Erreur fatale :", err);
  process.exit(1);
});
