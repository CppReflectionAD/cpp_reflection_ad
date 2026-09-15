#ifndef REFLECT_DEMO_COMPILE_TIME_UNORDERED_MAP_H
#define REFLECT_DEMO_COMPILE_TIME_UNORDERED_MAP_H

#include <cstddef>
#include <vector>

// A separately chained hash table usable during constant evaluation, where the
// standard unordered containers are not available. `HASH` is a default
// constructible callable mapping a `KEY` to a `std::size_t`; keys are compared
// with `operator==`.
template <class KEY, class VALUE, class HASH> class ConstevalHashMap {
public:
  // Return the address of the value stored under `key`, or nullptr if there is
  // no such entry. The pointer is invalidated by the next `insert`.
  consteval const VALUE *find(const KEY &key) const {
    const std::size_t h = HASH{}(key);
    const Bucket &bucket = buckets[h % buckets.size()];
    // Scan the bucket, comparing the cheap hash before the key itself.
    for (const Entry &entry : bucket) {
      if (entry.hash == h && entry.key == key) {
        return &entry.value;
      }
    }
    return nullptr;
  }

  // Store `value` under `key`. The caller is expected to have checked with
  // `find` that the key is absent.
  consteval void insert(const KEY &key, const VALUE &value) {
    const std::size_t h = HASH{}(key);
    buckets[h % buckets.size()].push_back(Entry{h, key, value});
    ++entryCount;
    grow_if_loaded();
  }

private:
  // The hash is kept alongside the key so that bucket scans do
  // not have to recompute it.
  struct Entry {
    std::size_t hash;
    KEY key;
    VALUE value;
  };
  using Bucket = std::vector<Entry>;
  static constexpr std::size_t kInitialBuckets = 64;
  // Grow once a bucket holds this many entries on average.
  static constexpr std::size_t kMaxLoadFactor = 2;

  std::vector<Bucket> buckets = std::vector<Bucket>(kInitialBuckets);
  std::size_t entryCount = 0;

  // Double the bucket count once the table is loaded past `kMaxLoadFactor`, so
  // the average bucket stays short and lookups stay constant time.
  consteval void grow_if_loaded() {
    if (entryCount <= buckets.size() * kMaxLoadFactor) {
      return;
    }
    std::vector<Bucket> fresh(buckets.size() * 2);
    for (const Bucket &bucket : buckets) {
      for (const Entry &entry : bucket) {
        fresh[entry.hash % fresh.size()].push_back(entry);
      }
    }
    buckets = fresh;
  }
};

#endif
