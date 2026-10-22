// Copies the version Changesets wrote to package.json into the Python package.
import { readFileSync, writeFileSync } from "node:fs";

const { version } = JSON.parse(readFileSync("package.json", "utf8"));

for (const [file, pattern] of [
  ["pyproject.toml", /^version = ".*"$/m],
  ["src/modelinfo/__init__.py", /^__version__ = ".*"$/m],
]) {
  const text = readFileSync(file, "utf8");
  if (!pattern.test(text)) throw new Error(`No version line found in ${file}`);
  writeFileSync(file, text.replace(pattern, (line) => line.replace(/".*"/, `"${version}"`)));
}

console.log(`Synced version ${version} to pyproject.toml and src/modelinfo/__init__.py`);
