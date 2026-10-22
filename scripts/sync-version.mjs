// Copies the version Changesets wrote to package.json into the Python package.
import { readFileSync, writeFileSync } from "node:fs";

const { version } = JSON.parse(readFileSync("package.json", "utf8"));

function bump(text, pattern, file) {
  if (!pattern.test(text)) throw new Error(`No version line found in ${file}`);
  return text.replace(pattern, (line) => line.replace(/".*"/, `"${version}"`));
}

writeFileSync(
  "pyproject.toml",
  bump(readFileSync("pyproject.toml", "utf8"), /^version = ".*"$/m, "pyproject.toml"),
);
writeFileSync(
  "src/modelinfo/__init__.py",
  bump(
    readFileSync("src/modelinfo/__init__.py", "utf8"),
    /^__version__ = ".*"$/m,
    "src/modelinfo/__init__.py",
  ),
);

console.log(`Synced version ${version} to pyproject.toml and src/modelinfo/__init__.py`);
