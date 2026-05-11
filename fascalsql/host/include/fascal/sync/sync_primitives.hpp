#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>

using namespace std;

namespace fascal {
namespace sync {

/// Atomic ticket counter for morsel-based execution.
/// Uses atomic operations to provide lock-free ticket allocation.
class TicketCounter {
 public:
  TicketCounter() : value_(0) {}

  /// Get next ticket number (atomic increment).
  int next() { return value_.fetch_add(1, memory_order_relaxed); }

  /// Reset counter to zero.
  void reset() { value_.store(0, memory_order_release); }

  /// Get current value without incrementing.
  int get() const { return value_.load(memory_order_acquire); }

 private:
  atomic<int> value_;
};

/// Bucket counter for GPU wakeup signaling.
/// Tracks completion of morsels in buckets and signals GPU when bucket is full.
class BucketCounter {
 public:
  BucketCounter(int num_buckets)
      : num_buckets_(num_buckets),
        counts_(num_buckets, 0),
        regular_bucket_size_(0),
        last_bucket_size_(0) {}

  /// Increment count for a bucket and signal GPU if bucket is full.
  void increment(int bucket_id) {
    lock_guard<mutex> lock(mutex_);
    counts_[bucket_id]++;
    if (is_bucket_full(bucket_id)) {
      signal_gpu(bucket_id);
    }
  }

  /// Check if a bucket is full (all morsels completed).
  bool is_bucket_full(int bucket_id) const {
    return counts_[bucket_id] >= bucket_size(bucket_id);
  }

  /// Wait for a bucket to be full (GPU-side).
  void wait_for_bucket(int bucket_id) {
    unique_lock<mutex> lock(mutex_);
    cond_.wait(lock, [this, bucket_id]() {
      return counts_[bucket_id] >= bucket_size(bucket_id);
    });
  }

  /// Set bucket sizes (regular and last bucket may differ).
  void set_bucket_sizes(int regular_size, int last_size) {
    lock_guard<mutex> lock(mutex_);
    regular_bucket_size_ = regular_size;
    last_bucket_size_ = last_size;
  }

  /// Get count for a bucket.
  int get_count(int bucket_id) const {
    lock_guard<mutex> lock(mutex_);
    return counts_[bucket_id];
  }

  /// Reset all counts.
  void reset() {
    lock_guard<mutex> lock(mutex_);
    fill(counts_.begin(), counts_.end(), 0);
  }

  /// Notify all waiting threads.
  void notify_all() {
    cond_.notify_all();
  }

 private:
  int num_buckets_;
  vector<int> counts_;
  int regular_bucket_size_{0};
  int last_bucket_size_{0};
  mutable mutex mutex_;
  condition_variable cond_;

  int bucket_size(int bucket_id) const {
    // Last bucket may be smaller
    if (bucket_id == num_buckets_ - 1) {
      return last_bucket_size_;
    }
    return regular_bucket_size_;
  }

  void signal_gpu(int bucket_id) {
    cond_.notify_all();
  }
};

/// Simple event flag for thread coordination.
class EventFlag {
 public:
  EventFlag() : flag_(false) {}

  /// Set flag to true.
  void set() { flag_.store(true, memory_order_release); }

  /// Reset flag to false.
  void reset() { flag_.store(false, memory_order_release); }

  /// Check if flag is set.
  bool is_set() const { return flag_.load(memory_order_acquire); }

  /// Wait for flag to be set.
  void wait() {
    while (!is_set()) {
      // Spin wait - could use condition_variable for longer waits
      __asm__ volatile("pause");
    }
  }

 private:
  atomic<bool> flag_;
};

/// Barrier for synchronizing multiple threads.
class SimpleBarrier {
 public:
  SimpleBarrier(int num_threads)
      : num_threads_(num_threads),
        count_(0),
        generation_(0) {}

  /// Wait for all threads to reach the barrier.
  void wait() {
    unique_lock<mutex> lock(mutex_);
    int gen = generation_.load(memory_order_acquire);

    int my_count = count_.fetch_add(1, memory_order_acq_rel);

    if (my_count == num_threads_) {
      // Last thread to arrive
      generation_.store(gen + 1, memory_order_acq_rel);
      count_.store(0, memory_order_release);
      cond_.notify_all();
    } else {
      // Wait for last thread
      while (generation_.load(memory_order_acquire) == gen) {
        cond_.wait(lock);
      }
    }
  }

 private:
  int num_threads_;
  atomic<int> count_;
  atomic<int> generation_;
  mutex mutex_;
  condition_variable cond_;
};

}  // namespace sync
}  // namespace fascal
