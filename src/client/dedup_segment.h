/* Creation and attachment of a dedup shared-memory segment (NodeDedup and
 * GpuNodeDedup).
 *
 * Exactly one process creates a segment: shm_open with O_EXCL. Before the
 * segment gets its size, the creator writes its pid into the header's
 * creator field, then sizes and backs the segment; the caller lays out the
 * header and stores 1 to init_done last. Every other process waits for
 * init_done.
 *
 * A creator killed before it finished (sizing and backing a 512 MiB segment
 * takes a while) used to leave a segment that every later process waited
 * on and then ran without. An opener that finds the segment unfinished once
 * its wait is over finishes the creation itself when the creator named in
 * it is dead, or when the segment is still empty (a creator killed between
 * its create and its pid write: those are a system call apart, not a whole
 * wait). The creator field is swapped from the dead pid (or 0) to its own,
 * so of several such openers one lays out the header and the others wait
 * for it. Laying a segment out again never rewinds anything a user of it
 * may depend on (the callers write only constant header fields).
 *
 * Left alone: a sized segment whose creator field is 0. That is an older
 * release's creator, which does not name itself and may still be backing
 * the segment; finishing it could be undone by that creator, which (older
 * releases) also resets the host ring cursor when it lays the header out.
 *
 * Liveness is a pid check: processes that share a segment must share a
 * PID namespace. */
#ifndef DFKV_DEDUP_SEGMENT_H_
#define DFKV_DEDUP_SEGMENT_H_

#include <sys/types.h>

#include <cstddef>
#include <string>

namespace dfkv {
namespace dedup_segment {

struct Layout {
  size_t len;            // whole segment
  size_t init_done_off;  // std::atomic<uint32_t>, 1 once laid out
  size_t creator_off;    // std::atomic<uint32_t>, pid of the creator
};

// Maps the segment. *creator: this process must lay out the header and then
// store 1 to init_done. nullptr: run without (logged, with tag as prefix).
// *hold_fd: the descriptor of the segment, holding a shared flock for as long
// as this process uses it; give it to Detach when done (-1 on failure). Not
// fork-safe after Attach: a forked child shares the open file description,
// and so the flock, of its parent (attach again in the child).
void* Attach(const std::string& name, const Layout& layout, const char* tag,
             bool* creator, int* hold_fd);

// Releases what Attach holds. The last attacher removes the segment's name
// (it took a 516 MiB segment for good, one more per namespace). An older
// release takes no lock, so one of its processes can be left on a segment
// removed under it: still correct, no longer shared, until it exits.
void Detach(const std::string& name, int hold_fd);

// What a creator killed right after naming itself (and, with sized, after
// sizing the segment) leaves behind. For tests.
bool AbandonForTest(const std::string& name, const Layout& layout,
                    pid_t creator, bool sized);

}  // namespace dedup_segment
}  // namespace dfkv

#endif  // DFKV_DEDUP_SEGMENT_H_
