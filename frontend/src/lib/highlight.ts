import type { Element, ElementContent, Root } from "hast";
import bash from "highlight.js/lib/languages/bash";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import yaml from "highlight.js/lib/languages/yaml";
import { createLowlight } from "lowlight";

// Why not rehype-highlight: it always bundles highlight.js's ~40 "common"
// languages (about 400 kB of the bundle) even when told to use fewer. A
// report's code blocks are SQL the data agent ran, the odd snippet of Python
// or JSON, so these few are registered and the rest never ship.
const lowlight = createLowlight({ bash, javascript, json, python, sql, typescript, yaml });
lowlight.registerAlias({ bash: ["sh", "shell", "console"], javascript: ["js"], typescript: ["ts"], python: ["py"], yaml: ["yml"] });

function text(node: ElementContent): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(text).join("");
  return "";
}

function visit(node: Root | Element) {
  for (const child of node.children) {
    if (child.type !== "element") continue;
    const code = child.tagName === "pre" ? child.children.find((c): c is Element => c.type === "element" && c.tagName === "code") : undefined;
    if (!code) {
      visit(child);
      continue;
    }
    // Only fenced blocks that name a language: guessing (highlightAuto) is
    // slow and often wrong on short snippets.
    const classes = Array.isArray(code.properties.className) ? code.properties.className.map(String) : [];
    const lang = classes.find((c) => c.startsWith("language-"))?.slice("language-".length);
    if (!lang || !lowlight.registered(lang)) continue;
    const result = lowlight.highlight(lang, code.children.map(text).join(""));
    code.properties.className = ["hljs", ...classes];
    code.children = result.children as ElementContent[];
  }
}

/** Rehype plugin: syntax-highlight ```lang fenced code blocks (highlight.js classes). */
export function rehypeCodeHighlight() {
  return (tree: Root) => visit(tree);
}
