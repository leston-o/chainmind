// Shared NEAR connection helpers. Keys come from the standard near-cli
// keystore at ~/.near-credentials/<network>/<account>.json.

import { connect, keyStores, providers } from "near-api-js";
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, existsSync } from "node:fs";

export const NETWORK = process.env.NEAR_NETWORK ?? "testnet";
export const RPC =
  process.env.NEAR_RPC ?? (NETWORK === "mainnet" ? "https://rpc.mainnet.near.org" : "https://rpc.testnet.near.org");
export const EXPLORER = NETWORK === "mainnet" ? "https://nearblocks.io" : "https://testnet.nearblocks.io";
export const TGAS = 10n ** 12n;
export const MAX_GAS = 300n * TGAS;

export const keyStore = new keyStores.UnencryptedFileSystemKeyStore(join(homedir(), ".near-credentials"));

export async function nearConnection() {
  return connect({ networkId: NETWORK, nodeUrl: RPC, keyStore });
}

export function deployment() {
  const path = new URL("../deployment.json", import.meta.url);
  if (!existsSync(path)) throw new Error("no deployment.json yet: run `npm run deploy` first");
  return JSON.parse(readFileSync(path, "utf8"));
}

export const result = (outcome) => providers.getTransactionLastResult(outcome);

export const txUrl = (outcome) => `${EXPLORER}/txns/${outcome.transaction.hash}`;

export function totalTgas(outcome) {
  const burnt =
    outcome.transaction_outcome.outcome.gas_burnt +
    outcome.receipts_outcome.reduce((s, r) => s + r.outcome.gas_burnt, 0);
  return (burnt / 1e12).toFixed(1);
}
