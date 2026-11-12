// Ask the onchain model to write, as a transaction every validator executes.
//
//   node generate.mjs "<prompt>" --as <your-account.testnet> [--max 48] [--temp 0.8] [--seed 42]
//   node generate.mjs "<prompt>" --view --seed 42      (free, read-only RPC call)
//   node generate.mjs --story --as <you> ["a nudge"]   (extend the Endless Story)

import { deployment, nearConnection, MAX_GAS, txUrl, result, totalTgas } from "./near.mjs";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const positional = args.filter((a, i) => !a.startsWith("--") && !(i > 0 && args[i - 1].startsWith("--") && !["--view", "--story"].includes(args[i - 1])));
const { contract_id } = deployment();
const near = await nearConnection();

if (args.includes("--view")) {
  const viewer = await near.account(contract_id);
  const g = await viewer.viewFunction({
    contractId: contract_id,
    methodName: "complete",
    args: {
      prompt: positional[0] ?? "Once upon a time",
      max_tokens: Number(opt("--max", 48)),
      temperature: Number(opt("--temp", 0.8)),
      seed: String(opt("--seed", 0)),
    },
  });
  console.log(g.prompt + g.text);
  console.log(`\n[${g.new_tokens} tokens, ${g.gas_burnt_tgas.toFixed(1)} Tgas, stop=${g.stop}, seed=${g.seed}]`);
  process.exit(0);
}

const as = opt("--as");
if (!as) {
  console.error("pass --as <your-account> to sign the transaction (or --view for a free read-only call)");
  process.exit(2);
}
const signer = await near.account(as);

if (args.includes("--story")) {
  const o = await signer.functionCall({
    contractId: contract_id,
    methodName: "extend_story",
    args: { nudge: positional[0] ?? null },
    gas: MAX_GAS,
  });
  const ch = result(o);
  console.log(`chapter ${ch.index} by ${ch.author}:\n${ch.text}`);
  console.log(`\n[${totalTgas(o)} Tgas]  ${txUrl(o)}`);
  process.exit(0);
}

const seed = opt("--seed");
const o = await signer.functionCall({
  contractId: contract_id,
  methodName: "generate",
  args: {
    prompt: positional[0] ?? "Once upon a time",
    max_tokens: Number(opt("--max", 48)),
    temperature: Number(opt("--temp", 0.8)),
    seed: seed === undefined ? null : String(seed),
  },
  gas: MAX_GAS,
});
const g = result(o);
console.log(g.prompt + g.text);
console.log(`\n[${g.new_tokens} tokens, stop=${g.stop}, seed=${g.seed}, ${totalTgas(o)} Tgas]`);
console.log(`every validator ran this: ${txUrl(o)}`);
console.log(`re-run it yourself: cargo run -p chainmind-tools --release --bin chainmind -- run models/stories260k.cmnd ${JSON.stringify(g.prompt)} --max ${opt("--max", 48)} --temp ${g.temperature} --seed ${g.seed}`);
