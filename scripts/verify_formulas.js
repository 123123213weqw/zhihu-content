#!/usr/bin/env node
/* Verify every LaTeX formula in a markdown article renders under KaTeX.
 * Usage: node verify_formulas.js <article.md>
 * Exit 0 if all formulas render; otherwise prints each failure with reason. */

const fs = require("fs");
const katex = require("katex");

const path = process.argv[2];
if (!path) { console.error("usage: node verify_formulas.js <article.md>"); process.exit(2); }

const md = fs.readFileSync(path, "utf8");
// strip code fences so $$ inside code samples are not treated as formulas
const stripped = md.replace(/```[\s\S]*?```/g, "");

// extract $$block$$ first, then $inline$ on what remains
const formulas = [];
const noBlocks = stripped.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => { formulas.push(tex.trim()); return ""; });
noBlocks.replace(/\$([^$\n]+?)\$/g, (_, tex) => { formulas.push(tex.trim()); return ""; });

let fail = 0;
formulas.forEach((tex, i) => {
  try {
    katex.renderToString(tex, { throwOnError: true, strict: false, displayMode: true });
  } catch (e) {
    fail++;
    console.log(`FAIL [${i}] ${tex.slice(0, 90)}`);
    console.log(`      -> ${String(e.message).slice(0, 140)}`);
  }
});
console.log(`\n${formulas.length} formulas checked, ${fail} failed`);
process.exit(fail ? 1 : 0);
