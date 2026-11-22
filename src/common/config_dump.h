/* Startup effective-config recorder.
 *
 * Goal: from the log alone, see EVERY parameter a dfkv process is actually
 * running with — flag, env var, or built-in default — instead of guessing which
 * of the ~40 DFKV_* env knobs took effect on a given node.
 *
 * Model: components record each parameter as they resolve it (the Env* helpers
 * both read the env AND record the resolved value + its source), and the process
 * entrypoint (server / mds main, or the KVClient ctor for the client) calls
 * Emit() once startup config has settled. Emit() also scans `environ` so any
 * DFKV_* env var that is set but not explicitly recorded still shows up — a knob
 * is never silently missing. Explicitly-recorded entries win over the scan (they
 * carry the resolved value, incl. defaults, and the precise source).
 *
 * Not on the hot path; all state is process-global and mutex-guarded.
 */
#ifndef DFKV_CONFIG_DUMP_H_
#define DFKV_CONFIG_DUMP_H_

#include <cstdint>
#include <string>

namespace dfkv {
namespace config_dump {

// Where a resolved parameter came from. kFlag/kEnv/kArg are all EXTERNALLY
// specified (CLI flag / env var / caller-passed API arg); kDefault is the
// process's built-in default (the operator set nothing).
enum class Source { kFlag, kEnv, kArg, kDefault };

// Record a resolved parameter (last write wins per name). Thread-safe.
void Record(const std::string& name, const std::string& value, Source src);

// Record an already-computed EFFECTIVE value for env knob `name`; the source is
// env when `name` is present in the environment, else default. For sites that
// keep their own (clamping/validating) parse and just want the true effective
// value in the dump — one line, no parse duplication, no drift.
void RecordResolved(const std::string& name, const std::string& value);

// Env helpers: read env `name`, record (value + kEnv/kDefault source), and
// return the resolved value. An unset OR empty env string yields `def` (the
// `e && *e` convention that dominates the codebase). Numeric helpers parse
// with utils/env_parse.h, which keeps what these helpers always did with an
// unclean value and adds a WARN: a number followed by other characters keeps
// the number, a negative EnvU64 wraps, anything else falls back to `def`.
std::string EnvStr(const char* name, const std::string& def = "");
int64_t     EnvI64(const char* name, int64_t def);
uint64_t    EnvU64(const char* name, uint64_t def);
// 1/on/true/yes and 0/off/false/no in any case; any other word reads as on
// (as it always did), with a WARN.
bool        EnvBool(const char* name, bool def);

// Log all records at INFO under a `<section>` header (one line per parameter,
// `name = value (source)`, sorted by name), folding in any un-recorded DFKV_*
// env vars from `environ`, then clear the registry. No-op if nothing to emit.
void Emit(const std::string& section);

// Test/reset hook: drop all records without emitting.
void ResetForTest();

}  // namespace config_dump
}  // namespace dfkv

#endif  // DFKV_CONFIG_DUMP_H_
