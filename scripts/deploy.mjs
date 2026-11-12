// Deploy Chainmind to a fresh sub-account, upload the model, seal it.
//
//   node deploy.mjs <your-account.testnet> [--model ../models/stories260k.cmnd] [--fund 10] [--lock]
//
// Needs your account's key in ~/.near-credentials/testnet/ (near-cli puts it
// there on login). --lock deletes the sub-account's only full-access key after
// sealing, so nobody, including you, can ever change the code or the model.

import { KeyPair, utils } from "near-api-js";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { keyStore, nearConnection, NETWORK, MAX_GAS, TGAS, txUrl, result, totalTgas } from "./near.mjs";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const ownerId = args[0];
if (!ownerId || ownerId.startsWith("--")) {
  console.error("usage: node deploy.mjs <your-account.testnet> [--model path] [--fund NEAR] [--lock]");
  process.exit(2);
}
const modelPath = opt("--model", "../models/stories260k.cmnd");
const wasmPath = opt("--wasm", "../target/wasm32-unknown-unknown/release/chainmind.wasm");
const fund = opt("--fund", "10");
const lock = args.includes("--lock");
const CHUNK = 200 * 1024;

const blob = readFileSync(modelPath);
const wasm = readFileSync(wasmPath);
const sha = createHash("sha256").update(blob).digest("hex");
console.log(`model: ${modelPath} (${blob.length} bytes, sha256 ${sha})`);
console.log(`contract: ${wasmPath} (${wasm.length} bytes)`);

const near = await nearConnection();
const owner = await near.account(ownerId);
const mindId = opt("--account", `mind-${Date.now().toString(36)}.${ownerId}`);

const kp = KeyPair.fromRandom("ed25519");
console.log(`\ncreating ${mindId} with ${fund} NEAR…`);
await owner.createAccount(mindId, kp.getPublicKey(), utils.format.parseNearAmount(fund));
await keyStore.setKey(NETWORK, mindId, kp);
const mind = await near.account(mindId);

console.log("deploying contract…");
const d = await mind.deployContract(wasm);
console.log(`  ${txUrl(d)}`);
await mind.functionCall({ contractId: mindId, methodName: "new", args: { owner_id: mindId }, gas: 30n * TGAS });

const txs = [];
for (let i = 0; i * CHUNK < blob.length; i++) {
  const data = blob.subarray(i * CHUNK, (i + 1) * CHUNK).toString("base64");
  const o = await mind.functionCall({
    contractId: mindId,
    methodName: "upload_chunk",
    args: { index: i, data },
    gas: MAX_GAS,
  });
  console.log(`upload chunk ${i}: ${totalTgas(o)} Tgas  ${txUrl(o)}`);
  txs.push(o.transaction.hash);
}

const s = await mind.functionCall({
  contractId: mindId,
  methodName: "seal",
  args: { expected_sha256: sha },
  gas: MAX_GAS,
});
const info = result(s);
console.log(`sealed: ${info.params} parameters, sha256 ${info.sha256}\n  ${txUrl(s)}`);

if (lock) {
  await mind.deleteKey(kp.getPublicKey());
  console.log("locked: full-access key deleted. The code and model can never change.");
}

const out = { network: NETWORK, contract_id: mindId, model_sha256: sha, model_bytes: blob.length, locked: lock, seal_tx: s.transaction.hash };
writeFileSync(new URL("../deployment.json", import.meta.url), JSON.stringify(out, null, 2) + "\n");
console.log(`\nwrote deployment.json. Try: npm run generate -- "Once upon a time"`);
