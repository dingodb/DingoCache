# Slab: fair eviction across size classes (STO-35)

Status: implemented behind `DFKV_SLAB_BALANCE` (default off), for the slab
disk store only; measured (section 10). The on-disk format does not change.
The RAM tier shares the allocator but does not call the balancer; it is a
follow-up.

## 1. The problem

A slab store gives each extent to one size class. When a class has no free
slot, the allocator mostly evicts from that class's own slots; extents move
between classes rarely. Space therefore stays with the classes that took it
first, whatever each class's data is worth now.

In a unit measurement (64 MiB store of 4 MiB extents, 64 KiB granularity;
three value sizes written in equal bytes, 140 MB in all), the share of the
last capacity's worth of writes still held at the end was:

| Class | Values held of the most recent writes |
|---|---|
| 60 KB (64 KiB slots) | 92% |
| 900 KB | 52% |
| 3 MB | 25% |

The small class kept old values while the large class lost new ones. The
allocator moved 16 extents in the run, and no cold steals happened
(`cold_steals=0`).

That harness exaggerates one cause and hides another (section 3.4). Measured
at a realistic extent count (section 10), steady demand is fair; the freeze
is real when the mix of sizes changes while every class is still written.

## 2. What the allocator does today

`SlabAllocator::Put` (slab_allocator.cc), for a key of class C:

1. Bind unbound extents while C has fewer than `stripe_ways` extents with
   free slots (8 on disk, 1 in the RAM tier).
2. Pop a free slot of C (the fast path).
3. `TakeEmptyExtentFor`: take an extent another class holds with no
   resident, if that class is idle or has more than `stripe_ways` such
   extents.
4. Floor steal: if C has fewer than `stripe_ways` extents, steal the least
   recently used extent of a class that has more.
5. `StealColdExtentFor`: steal another class's extent if:
   - its youngest resident is older than one cold window (default: the
     number of residents, in admissions); and
   - its residents' mean age is more than `kStealAgeMargin` (2) times that
     of C's own least recently used extent.
6. `EvictOneFrom(C)`: CLOCK second chance in C's ring, which is ordered by
   admission and actual reads.
7. Last resort: steal any extent.

Outside `Put`:
- The disk store's and the RAM tier's reclaimers keep free slots ahead of
  writers (`ReclaimClass`, `FreeSlotTarget`). They also grow a hot class
  from `ColdestGrowDonor`: a class whose write demand, in bytes, is at most
  a quarter of the hot class's.
- The watermark (`EvictColdToTarget`) evicts globally oldest slots across
  classes along a global ring. It only frees slots inside each class, and
  it only runs above the high watermark.

## 3. Why the shares freeze

### 3.1 The cold-steal gate never opens between busy classes

CLOCK picks victims across all of a class's extents, so its free slots, and
with them its new writes, are spread over every extent it holds. Every
extent of a class that is being written therefore holds a recent resident:
its `youngest_seq` is always inside the cold window, and step 5 never
applies.

### 3.2 The 2× mean-age margin compares unlike things

When step 5 does get past the gate, it compares a donor extent's mean
resident age (about half the donor class's residency time, since residents
are spread over it) with the target's own least recently used extent. A
donor class can hold its data about four times longer than the target
before an extent moves.

### 3.3 Background growth only looks at write demand

`ColdestGrowDonor` needs the donor's byte demand to be at most a quarter of
the hot class's. Classes with similar byte demand but different slot
packing never qualify. That is the equal-bytes case, and a common one: KV
pages and their side objects are written together.

### 3.4 The harness geometry

With 16 extents and `stripe_ways` 8:
- The first two classes each bind 8 extents at their first `Put` (step 1).
- The floor steal (step 4) needs a donor with more than 8 extents, which
  can no longer exist.

So the measured shares mostly reflect arrival order. At a realistic extent
count (about 1,000 per TiB at the default 1 GiB extent), steps 1 and 4 are
negligible, and 3.1 to 3.3 decide. The effect is expected there too, but
for those reasons; step 0 measures it.

## 4. Goal

Residency-time fairness: the age of what each class evicts next ("tail
age", on the allocator's operation clock `activity_seq_`) should be about
the same for every class that is short of space. This is what a single LRU
over all values would do, at the granularity the format allows:
- Space can only change class by whole extents. The existing
  `on_extent_bind` wipe already records a rebinding: the table wipe is
  written before any resident is forgotten (STO-11: kept by the page cache
  for a warm restart). Moving more often needs no format change.
- Reads count: `TouchLocked` and `PinForRead` update `access_seq`. A small
  class whose values are read keeps its space against a large class that is
  only written.

Non-goals: changing in-class victim choice (that is STO-18); changing the
watermark.

## 5. Options

**A. Global LRU per slot across classes.** Evict the globally oldest slot
whatever its class. This frees a slot in the wrong class: a 3 MB value
needs a 3 MB slot, and an extent only changes class when it is empty or
wiped. With CLOCK's scattering (3.1), evicting old slots across classes
rarely empties an extent. It does not move space. Rejected.

**B. Tail-age balancing by whole extents.** Move one extent from the class
with the oldest tail to the class with the youngest, when the ages differ
by more than a ratio. Memcached's slab page mover balances classes by
comparing the ages at the ends of their LRUs in a similar way. This is the
proposal.

**C. Extent-granular eviction inside a class.** A class evicts its least
recently used extent as a whole, as a log-structured store does with
segments. Extents then age uniformly, so moving one between classes costs
nothing extra. But every eviction drops a whole extent's residents,
including recently read ones, unless reads relocate them. That is a larger
change of in-class policy, and it belongs with STO-18 and the log engine's
experience, not here.

**D. Tune the existing rules.** For example a smaller `kStealAgeMargin`, a
looser cold window, or a demand ratio above a quarter. None of these fixes
3.1 (the gate is structural), and each loosening risks ping-pong between
classes.

## 6. Proposal (option B)

### 6.1 Tail age

A class's ring is in access order and its CLOCK hand sits at the oldest
resident, so:

    tail_age(c) = activity_seq_ - slots_[classes_[c].hand].access_seq

This is O(1) and needs no new state. A protected (retained) or pinned
resident at the hand is skipped by eviction. Using it overstates the tail
age by at most one resident's worth; the ratio absorbs that.

### 6.2 Where the decision runs

In the reclaim tick, not in `Put`:
- Both reclaimers (disk: `ReclaimTick`, every 50 ms; RAM tier: every 10 ms)
  already visit every class under the allocator lock in bounded steps.
- A non-empty steal evicts a whole extent's residents. A 1 GiB extent of
  64 KiB slots holds 16,384 of them, and doing that inside a `Put` would put
  milliseconds into that PUT's latency.

A new `SlabAllocator::BalanceTailAges(max_moves)`, called once per tick:

1. Among classes with residents and no free-slot headroom left (the classes
   evicting now), find the youngest tail Y and the oldest tail O, over
   classes that can give an extent (6.3).
2. If `tail_age(O) > R * tail_age(Y)` with R = 1.1 (section 10), move O's least
   recently used extent (by mean access age, the existing
   `LeastRecentlyUsedExtentLocked` / `StealsBeforeLocked` order, skipping
   pinned or protected extents) to Y through `StealExtentLocked`.
   - At most `max_moves` extents per tick (1).
   - A moved extent cannot move back for one cold window (hysteresis on top
     of R).
3. Count moves (`dfkv_slab_balance_moves_total`) and export each class's
   tail age as a gauge, so the effect is visible.

`StealColdExtentFor`'s `youngest_seq` gate in `Put` is left as it is. It
stays a fallback for idle donors, and the balancer makes it rarely needed.
`ColdestGrowDonor` (demand-based growth) also stays: it handles the case of
one class going idle, and the balancer handles classes that are all busy.

### 6.3 Who may give

A class gives no extent if:
- it would drop below its floor, `min(stripe_ways, max(1, bound_extents /
  (4 × classes evicting)))`. This keeps the striping width where the pool
  is large, and stops the floor from freezing a small pool (3.4);
- or it has no extent that is unpinned and unprotected.

### 6.4 Restart

Rebuilt residents have `access_seq` 0, so all tails start with equal ages
and nothing moves until classes have turned over part of their data. No
state is persisted.

### 6.5 Cost

| Path | Today | With the balancer |
|---|---|---|
| `Put` fast path (free slot) | pop | unchanged |
| `Put` slow path | steps 3–7 | unchanged |
| Reclaim tick | per-class reclaim | plus O(classes) tail ages; when moving, one O(extents) scan for the donor's LRU extent (about 1,000 entries per TiB, a few µs) and one `StealExtentLocked` |
| Steal itself | existing (extent wipe write, index removals) | same work, at most one per tick |

Lock hold per move is that of a steal today. With 1 GiB extents of small
slots it is dominated by index removals; the measurement reports the hold
time, and the tick can drop to every other tick if it matters.

## 7. Relation to STO-18 (chain-aware eviction)

STO-35 decides how much space each class has; STO-18 decides which values
inside that space go.

They meet in one place: a prefix chain often spans classes. Examples are a
KV page plus its side-pool objects, or the 1-byte anchor markers (STO-14).
If the large class evicts after a quarter of the time the small class keeps
values, the small objects of chains whose pages are gone are dead weight.
That is the 92% / 25% split above. Equal tail ages are a precondition for
chain-aware eviction to keep whole chains.

STO-18's offline replay harness (recorded key sequences replayed against
the allocator) is also the right tool to evaluate B on real traffic. This
design therefore keeps the balancer separate (one function, one tick hook),
so STO-18 can replace the tail-age measure with a chain-aware value later.

## 8. Measurement plan

**Harness A (allocator only, no I/O).** `SlabAllocator` is media-agnostic,
so a unit test can drive millions of `Put`s against a pool shaped like a
real disk (e.g. 1,024 extents with `stripe_ways` 8) with no disk writes,
and call `BalanceTailAges` as the tick would. It records:
- the share of the most recent capacity's worth of writes still held, per
  class;
- payload held / capacity;
- moves;
- the allocator's time per `Put` (ns) and the lock hold per tick with a
  move.

It runs before and after in the same test binary (the balancer on or off).

Scenarios:
- S0 (step 0): S1 on today's code, to confirm STO-35 at a realistic extent
  count.
- S1: 60 KB / 900 KB / 3 MB written in equal bytes. Expect the per-class
  retention to converge (within 10 points of each other), with total
  payload held not lower than today.
- S2: skewed demand (80% of bytes 3 MB, 20% 60 KB). This guards the case
  today's rules handle.
- S3: phase change (two capacities of 60 KB only, then 3 MB only). Measure
  the writes needed until the 3 MB class holds 50% of its recent writes.
- S4: a small class whose hot set is re-read. It must keep its space against
  a write-only large class (reads count).

**Harness B (disk store, confirmation).** The earlier
`DiskSlabStore`-level harness, with the geometry fixed: enough extents for
the floors (per 3.4), 64 KiB granularity, direct writes, about 100 MB per
scenario for S1 and S3, before and after. It reports:
- retention per class;
- PUT p50/p99;
- `put_busy`;
- the reclaim tick's maximum lock hold.

Footprint:
- Harness A writes nothing to disk; it uses up to a few hundred MB of RAM
  for the allocator metadata.
- Harness B writes about 400 MB in all (2 scenarios × 2 variants × 100 MB).
  That is under the 1 GB limit, so no approval is needed beyond this plan.

Acceptance:
- S1 converges.
- S2 and S4 are no worse than today.
- S3 adapts within about one capacity of writes.
- `Put` slow-path time and PUT p99 are within noise.
- The lock hold per move is reported, with a bound proposed from the
  numbers.

## 9. Open questions

- R (1.5) and the per-tick move budget: set them from S1/S3 in harness A.
- Should the RAM tier get the balancer at the same time? It shares the
  allocator, so the code is there either way. Its arena has 32 MiB extents
  and much shorter residency, so harness A should include a RAM-shaped pool
  before it is switched on there.
- Whether to also retire `StealColdExtentFor` once the balancer is
  measured. Kept for now (6.2).

## 10. Measurements

Harness A: the allocator alone, no I/O, 1,024 extents of 64 MiB with
64 KiB alignment and the disk store's reclaim tick replayed every 64 MiB
written; classes of 60 KB, 900 KB and 3 MB values. "Held" is the share of
the most recent capacity's worth of writes still stored, per class.

| Case | Without the balancer | With it (R 1.1, 1 move per tick) |
|---|---|---|
| Equal bytes, steady (S0) | 87.5 / 86.5 / 87.0 | the same (no move) |
| Equal bytes, no watermark | 87.8 / 86.0 / 87.3 | the same (no move) |
| Equal, then 80% of the bytes in 3 MB, +3 capacities | 100 / 100 / 72.1 (extents 178/244/602) | 86.8 / 86.8 / 86.8 (114/183/727) |
| The same without the watermark | 100 / 100 / 77.6 | 100 / 95.4 / 93.4 |
| Small values only, then 3 MB only, +3 capacities | 88.1 | 88.6 |
| As the 80% case, the small class re-reading a hot set | hot reads 100% hit; 100 / 100 / 70.2 | hot reads 100% hit; 83.2 / 83.2 / 83.2 |

- So the shares freeze only when the mix shifts while all classes stay
  busy: section 3.1's gate keeps Put's cold steal shut, and background
  growth stops once a class's own evictions keep its free slots topped up.
- The fair split for 80/10/10 bytes is about 106/129/789 extents (by each
  class's packing); with the balancer the classes settle at 114/183/727.
- R 1.25 and 1.5 with one move per tick stall short of equal; two or more
  moves per tick overshoot (the small class fell to 54% before recovering).
  R 1.1 with one move converged without overshoot.
- A move of a 64 MiB extent of 64 KiB slots held the allocator lock up to
  about 240 µs; of a 1 GiB extent of 128 KiB slots (8,192 values), 0.83 to
  0.91 ms.

Harness B: the disk store (32 MiB of 1 MiB extents, values scaled to
6 KB / 90 KB / 300 KB, the tick driven every 2 MiB written, about 400 MB
written in all). Equal bytes then 80% in 300 KB:

| | Without | With |
|---|---|---|
| Held after the shift | 100 / 100 / 40 | 55.8 / 75.7 / 63.3 |
| Extents | 8 / 8 / 16 | 3 / 4 / 25 (fair: about 4 / 3 / 25) |
| PUT p50 / p99 | 6.1 / 80 µs | 5.8 / 70 µs |
| Longest store-lock hold of a move | — | 61 µs |

With 32 extents one extent is a quarter of a small class's fair share, so
the held shares stay uneven; the extent counts match the fair split.
