/**
 * Helpers shared by the plugin test files.
 *
 * They live outside `lib/` on purpose: `sync.mjs` copies that directory into
 * every plugin, and nothing here belongs in a shipped plugin. Importing a
 * helper from a `*.test.mjs` file would also register that file's own tests a
 * second time in whichever runner picked it up.
 */

import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildPluginConfig } from "../lib/plugin-config.mjs";

/**
 * A built config resolved against nothing at all, for callers that only need
 * the shape of the object a loader receives.
 */
export function buildConfigForTest(harness) {
  const dir = join(tmpdir(), "ov-plugin-config-absent");
  return buildPluginConfig(harness, {
    cwd: dir,
    env: {
      OPENVIKING_CLI_CONFIG_FILE: join(dir, "ovcli.conf"),
      OPENVIKING_CONFIG_FILE: join(dir, "ov.conf"),
      OPENVIKING_HOME: dir,
    },
  });
}
