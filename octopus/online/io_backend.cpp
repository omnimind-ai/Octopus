#include <torch/extension.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <liburing.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using Request = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
using Plan = std::vector<std::vector<Request>>;
using Clock = std::chrono::steady_clock;
constexpr unsigned QUEUE_DEPTH = 4096;
constexpr size_t MAX_CACHED_PINNED = 64 * 1024 * 1024;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stats {
    std::atomic<uint64_t> rings{0}, pinned_allocations{0}, file_opens{0}, threads{0};
    std::atomic<uint64_t> reads{0}, read_bytes{0}, read_jobs{0}, write_jobs{0};
    std::atomic<uint64_t> registrations{0}, registration_failures{0}, fixed_reads{0};
    std::map<std::string, uint64_t> snapshot() const {
        return {{"rings_created", rings.load()}, {"pinned_allocations", pinned_allocations.load()},
                {"file_opens", file_opens.load()}, {"worker_threads_created", threads.load()},
                {"submitted_reads", reads.load()}, {"read_bytes", read_bytes.load()},
                {"read_jobs", read_jobs.load()}, {"write_jobs", write_jobs.load()},
                {"buffer_registrations", registrations.load()}, {"buffer_registration_failures", registration_failures.load()},
                {"fixed_buffer_reads", fixed_reads.load()}};
    }
};

class Executor {
    std::mutex mu;
    std::condition_variable ready, idle;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    size_t active = 0;
    bool stopping = false;
public:
    explicit Executor(unsigned count, Stats& stats) {
        try {
            for (unsigned i = 0; i < count; ++i) {
                workers.emplace_back([this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(mu);
                            ready.wait(lock, [&] { return stopping || !queue.empty(); });
                            if (queue.empty()) return;
                            task = std::move(queue.front());
                            queue.pop_front();
                            ++active;
                        }
                        task(); // Submitted tasks always complete their promise, including on failure.
                        task = nullptr;
                        { std::lock_guard<std::mutex> lock(mu); --active; }
                        idle.notify_all();
                    }
                });
                ++stats.threads;
            }
        } catch (...) { stop(); throw; }
    }
    ~Executor() { stop(); }
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (stopping) throw std::runtime_error("I/O executor is closed");
            queue.push_back(std::move(task));
        }
        ready.notify_one();
    }
    void wait_idle() {
        std::unique_lock<std::mutex> lock(mu);
        idle.wait(lock, [&] { return queue.empty() && active == 0; });
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mu); stopping = true; }
        ready.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        workers.clear();
    }
};

struct File {
    std::string path;
    std::mutex mu;
    int read_fd = -1, write_fd = -1;
    uint64_t offline_size = 0;
    explicit File(std::string name) : path(std::move(name)) {}
    ~File() { if (read_fd >= 0) close(read_fd); if (write_fd >= 0) close(write_fd); }
    static uint64_t size(int fd) {
        struct stat info{};
        if (fstat(fd, &info) || !S_ISREG(info.st_mode)) throw std::runtime_error("KV path must be a regular file");
        return info.st_size;
    }
    int open_checked(int flags, Stats& stats) {
        int fd = open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) throw std::runtime_error("Cannot open KV file: " + path + ": " + strerror(errno));
        try { size(fd); } catch (...) { close(fd); throw; }
        ++stats.file_opens;
        return fd;
    }
    std::pair<int, uint64_t> reader(bool append, Stats& stats) {
        std::lock_guard<std::mutex> lock(mu);
        if (read_fd < 0) {
            read_fd = open_checked(O_RDONLY | O_DIRECT, stats);
            offline_size = size(read_fd);
        }
        // Offline files stay immutable until the next storage configuration.
        return {read_fd, append ? size(read_fd) : offline_size};
    }
    int writer(Stats& stats) {
        std::lock_guard<std::mutex> lock(mu);
        if (write_fd < 0) write_fd = open_checked(O_WRONLY | O_CREAT, stats);
        return write_fd;
    }
};
struct Storage {
    int64_t kv_heads, head_dim;
    size_t payload_bytes, block_bytes;
    std::vector<std::unique_ptr<File>> data, append;
    std::shared_future<void> last_write;
    bool write_error_reported = false;
    // Only the serial append worker accesses this error.
    std::exception_ptr write_error;
    Storage(const std::vector<std::string>& paths, const std::vector<std::string>& appends,
            int64_t heads, int64_t dim) : kv_heads(heads), head_dim(dim) {
        if (heads < 1 || dim < 1 || heads > 262144 || dim > 262144 || heads * dim > 262144)
            throw std::runtime_error("Invalid KV geometry: one BF16 KV record must fit in 1 MiB");
        payload_bytes = 4 * heads * dim;
        block_bytes = (payload_bytes + 4095) / 4096 * 4096;
        if (paths.empty() || paths.size() != appends.size()) throw std::runtime_error("Supply matching data/append paths");
        auto all = paths;
        all.insert(all.end(), appends.begin(), appends.end());
        std::sort(all.begin(), all.end());
        if (all.front().empty() || std::adjacent_find(all.begin(), all.end()) != all.end())
            throw std::runtime_error("Data and append paths must all be distinct and nonempty");
        for (const auto& path : paths) data.push_back(std::make_unique<File>(path));
        for (const auto& path : appends) append.push_back(std::make_unique<File>(path));
    }
    File& file(int64_t disk, int64_t is_append) {
        if (disk < 1 || disk > static_cast<int64_t>(data.size()) || (is_append != 0 && is_append != 1))
            throw std::runtime_error("Invalid disk ID or append flag");
        return *(is_append ? append : data)[disk - 1];
    }
};
struct Op {
    File* file;
    bool append;
    size_t offset, length, destination;
};
struct Prepared {
    std::shared_ptr<Storage> storage;
    std::shared_future<void> writes_ready;
    std::vector<Op> operations;
    std::vector<int64_t> tokens;
    size_t bytes = 0;
};
Prepared prepare(const Plan& input, std::shared_ptr<Storage> storage) {
    Prepared p;
    p.storage = std::move(storage);
    std::vector<std::vector<Op>> buckets;
    size_t maximum = 0;
    for (auto bucket : input) {
        if (bucket.empty()) continue;
        if (!p.storage) throw std::runtime_error("Storage not configured; supply --disk_paths");
        const size_t block_bytes = p.storage->block_bytes;
        std::sort(bucket.begin(), bucket.end(), [](const auto& a, const auto& b) {
            return std::tie(std::get<1>(a), std::get<0>(a)) < std::tie(std::get<1>(b), std::get<0>(b));
        });
        const auto disk = std::get<3>(bucket.front()), append = std::get<4>(bucket.front());
        File* file = &p.storage->file(disk, append);
        if (append) p.writes_ready = p.storage->last_write;
        for (const auto& r : bucket) {
            if (std::get<0>(r) < 0 || std::get<1>(r) < 0 || std::get<1>(r) % block_bytes ||
                std::get<1>(r) > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(block_bytes) ||
                std::get<2>(r) != block_bytes || std::get<3>(r) != disk || std::get<4>(r) != append)
                throw std::runtime_error("Expected aligned KV records grouped by disk and file");
            p.tokens.push_back(std::get<0>(r));
        }
        std::vector<Op> operations;
        for (size_t i = 0; i < bucket.size();) {
            const size_t offset = std::get<1>(bucket[i]);
            size_t bytes = block_bytes, end = i + 1;
            while (end < bucket.size() && std::get<1>(bucket[end]) == static_cast<int64_t>(offset + bytes) && bytes + block_bytes <= 1024*1024) {
                bytes += block_bytes;
                ++end;
            }
            operations.push_back({file, append != 0, offset, bytes, p.bytes});
            p.bytes += bytes;
            i = end;
        }
        maximum = std::max(maximum, operations.size());
        buckets.push_back(std::move(operations));
    }
    // Give every disk a share of each queue-depth wave, but group its SQEs
    // within the wave so the kernel can batch dispatch to the same device.
    const size_t quota = std::max<size_t>(1, QUEUE_DEPTH / std::max<size_t>(1, buckets.size()));
    for (size_t start = 0; start < maximum; start += quota) {
        for (const auto& bucket : buckets) {
            const size_t stop = std::min(start + quota, bucket.size());
            if (start < stop) p.operations.insert(p.operations.end(), bucket.begin() + start, bucket.begin() + stop);
        }
    }
    return p;
}

struct Pinned {
    void* allocation = nullptr;
    void* data = nullptr;
    size_t capacity;
    Pinned(size_t bytes, Stats& stats) : capacity(bytes) {
        cuda_check(cudaHostAlloc(&allocation, bytes + 4096, cudaHostAllocDefault));
        data = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(allocation) + 4095) & ~uintptr_t(4095));
        ++stats.pinned_allocations;
    }
    ~Pinned() { if (allocation) cudaFreeHost(allocation); }
};
struct Reader {
    std::mutex execution_mu;
    c10::cuda::CUDAStream stream;
    io_uring ring{};
    bool live = false, failed = false, registered = false;
    std::unique_ptr<Pinned> pinned;
    explicit Reader(int device, Stats& stats) : stream(c10::cuda::getStreamFromPool(false, device)) {
        int status = io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
        if (status < 0) throw std::runtime_error("io_uring_queue_init: " + std::string(strerror(-status)));
        live = true;
        if (!ring.sq.ring_entries) {
            ring.sq.ring_entries = *ring.sq.kring_entries;
            ring.sq.ring_mask = *ring.sq.kring_mask;
        }
        if (!ring.cq.ring_entries) {
            ring.cq.ring_entries = *ring.cq.kring_entries;
            ring.cq.ring_mask = *ring.cq.kring_mask;
        }
        ++stats.rings;
    }
    ~Reader() { if (live) io_uring_queue_exit(&ring); }
    void* buffer(size_t bytes, Stats& stats) {
        if (failed) throw std::runtime_error("I/O worker unavailable after submission queue failure");
        if (bytes > MAX_CACHED_PINNED) throw std::runtime_error("I/O tile exceeds pinned buffer capacity");
        if (!pinned || pinned->capacity < bytes) {
            size_t capacity = 64 * 1024;
            while (capacity < bytes) capacity *= 2;
            if (registered) {
                int status = io_uring_unregister_buffers(&ring);
                if (status < 0) throw std::runtime_error("Could not unregister I/O buffer");
                registered = false;
            }
            pinned.reset();
            pinned = std::make_unique<Pinned>(capacity, stats);
            iovec buffer{pinned->data, pinned->capacity};
            registered = io_uring_register_buffers(&ring, &buffer, 1) == 0;
            if (registered) ++stats.registrations;
            else ++stats.registration_failures; // Ordinary O_DIRECT reads remain available.
        }
        return pinned->data;
    }
    double read(const Prepared& p, void* host, Stats& stats) {
        if (p.writes_ready.valid()) p.writes_ready.get();
        std::unordered_map<File*, int> descriptors;
        for (const auto& op : p.operations) {
            if (!descriptors.count(op.file)) {
                auto entry = op.file->reader(op.append, stats);
                descriptors[op.file] = entry.first;
                for (const auto& check : p.operations) if (check.file == op.file)
                    if (check.offset > entry.second || check.length > entry.second - check.offset)
                        throw std::runtime_error("KV read exceeds file size: " + op.file->path);
            }
        }
        auto start = Clock::now();
        const bool fixed = registered && pinned && host == pinned->data;
        size_t next = 0, inflight = 0, done = 0;
        bool bad = false;
        try {
            while (done < p.operations.size()) {
                unsigned queued = 0;
                while (!bad && next < p.operations.size() && inflight + queued < QUEUE_DEPTH) {
                    const auto& op = p.operations[next];
                    auto* sqe = io_uring_get_sqe(&ring);
                    if (!sqe) throw std::runtime_error("Unexpected io_uring queue overflow");
                    auto* destination = static_cast<char*>(host) + op.destination;
                    if (fixed) io_uring_prep_read_fixed(sqe, descriptors.at(op.file), destination, op.length, op.offset, 0);
                    else io_uring_prep_read(sqe, descriptors.at(op.file), destination, op.length, op.offset);
                    io_uring_sqe_set_data64(sqe, next++);
                    ++queued;
                }
                while (queued) {
                    int n = io_uring_submit(&ring);
                    if (n == -EINTR) continue;
                    if (n <= 0) throw std::runtime_error("io_uring_submit failed or made no progress");
                    queued -= n;
                    inflight += n;
                    stats.reads += n;
                    if (fixed) stats.fixed_reads += n;
                }
                if (!inflight) break;
                io_uring_cqe* completed[256];
                unsigned count = 0, head;
                io_uring_cqe* entry;
                io_uring_for_each_cqe(&ring, head, entry) {
                    completed[count++] = entry;
                    if (count == 256) break;
                }
                if (!count) {
                    io_uring_cqe* cqe = nullptr;
                    int status;
                    do { status = io_uring_wait_cqe(&ring, &cqe); } while (status == -EINTR);
                    if (status < 0) throw std::runtime_error("io_uring_wait_cqe failed");
                    continue;
                }
                for (unsigned i = 0; i < count; ++i) {
                    auto index = io_uring_cqe_get_data64(completed[i]);
                    if (index >= p.operations.size() || completed[i]->res != static_cast<int>(p.operations[index].length)) bad = true;
                }
                io_uring_cq_advance(&ring, count);
                inflight -= count;
                done += count;
            }
            if (bad) throw std::runtime_error("Failed or short SSD read; inference stopped");
        } catch (...) {
            // Ordinary I/O errors have been fully drained. An unrecoverable ring
            // API error quarantines its buffer instead of reusing pages still in flight.
            if (inflight || io_uring_sq_ready(&ring)) failed = true;
            throw;
        }
        stats.read_bytes += p.bytes;
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
};
struct ReadJob {
    Prepared plan;
    torch::Tensor buffer;
    std::promise<void> completion;
    std::shared_future<void> ready = completion.get_future().share();
};
struct AppendTask { File* file; size_t offset; torch::Tensor data; };

class Backend {
    Stats stats;
    std::mutex api_mu, debug_mu;
    std::shared_ptr<Storage> storage;
    std::unique_ptr<Executor> readers, writer;
    std::unordered_map<int, std::shared_ptr<Reader>> synchronous_readers;
    std::unordered_map<int, std::shared_ptr<ReadJob>> requests;
    int next_handle = 1;
    std::atomic<bool> debug{false};
    std::string log_path;

    void log_read(size_t bytes, double seconds) {
        if (!debug.load()) return;
        std::lock_guard<std::mutex> lock(debug_mu);
        if (!log_path.empty()) std::ofstream(log_path, std::ios::app)
            << "[io_backend] pure_read_io bytes=" << bytes << " io_s=" << seconds << '\n';
    }
    void run(const std::shared_ptr<ReadJob>& job, int device, std::shared_ptr<Reader> synchronous = nullptr) {
        try {
            cuda_check(cudaSetDevice(device));
            // Destroyed on worker exit; CUDA streams themselves belong to PyTorch's pool.
            thread_local std::unordered_map<int, std::unique_ptr<Reader>> contexts;
            Reader* reader = synchronous.get();
            if (!reader) {
                auto& context = contexts[device];
                if (!context) context = std::make_unique<Reader>(device, stats);
                reader = context.get();
            }
            std::lock_guard<std::mutex> execution_lock(reader->execution_mu);
            c10::cuda::CUDAStreamGuard guard(reader->stream);
            job->buffer = torch::empty({static_cast<int64_t>(job->plan.tokens.size()),
                static_cast<int64_t>(job->plan.storage->block_bytes / 2)},
                torch::TensorOptions().dtype(torch::kBFloat16).device(torch::Device(torch::kCUDA, device)));
            const size_t capacity = std::min(job->plan.bytes, MAX_CACHED_PINNED);
            void* host = reader->buffer(capacity, stats);
            try {
                double seconds = 0;
                if (job->plan.bytes <= capacity) {
                    seconds = reader->read(job->plan, host, stats);
                    cuda_check(cudaMemcpyAsync(job->buffer.data_ptr(), host, job->plan.bytes, cudaMemcpyHostToDevice, reader->stream));
                    cuda_check(cudaStreamSynchronize(reader->stream));
                } else {
                    struct Copy { size_t host, gpu, bytes; };
                    size_t next = 0;
                    while (next < job->plan.operations.size()) {
                        Prepared tile;
                        tile.storage = job->plan.storage;
                        tile.writes_ready = job->plan.writes_ready;
                        std::vector<Copy> copies;
                        while (next < job->plan.operations.size()) {
                            auto op = job->plan.operations[next];
                            if (tile.bytes + op.length > capacity) break;
                            if (!copies.empty() && copies.back().gpu + copies.back().bytes == op.destination)
                                copies.back().bytes += op.length;
                            else copies.push_back({tile.bytes, op.destination, op.length});
                            op.destination = tile.bytes;
                            tile.bytes += op.length;
                            tile.operations.push_back(op);
                            ++next;
                        }
                        seconds += reader->read(tile, host, stats);
                        for (const auto& copy : copies)
                            cuda_check(cudaMemcpyAsync(static_cast<char*>(job->buffer.data_ptr()) + copy.gpu,
                                static_cast<char*>(host) + copy.host, copy.bytes, cudaMemcpyHostToDevice, reader->stream));
                        // The next tile reuses these host pages after DMA has finished.
                        cuda_check(cudaStreamSynchronize(reader->stream));
                    }
                }
                log_read(job->plan.bytes, seconds);
            } catch (...) {
                if (reader->failed) {
                    reader->pinned->allocation = nullptr;
                } else {
                    // A failed CUDA copy must not race reuse of the pinned buffer.
                    cudaStreamSynchronize(reader->stream);
                }
                throw;
            }
            ++stats.read_jobs;
            job->completion.set_value();
        } catch (...) { job->completion.set_exception(std::current_exception()); }
    }
    std::exception_ptr drain_locked() {
        if (writer) writer->wait_idle();
        if (readers) readers->wait_idle();
        std::exception_ptr error;
        for (auto& item : requests) {
            try { item.second->ready.get(); } catch (...) { if (!error) error = std::current_exception(); }
        }
        requests.clear();
        if (storage && storage->last_write.valid()) {
            try { storage->last_write.get(); } catch (...) {
                if (!error && !storage->write_error_reported) error = std::current_exception();
                storage->write_error_reported = true;
            }
        }
        return error;
    }
public:
    ~Backend() { try { close(); } catch (...) {} }
    void set_debug(bool enabled, const std::string& path) {
        std::lock_guard<std::mutex> lock(debug_mu);
        log_path = path;
        debug = enabled;
    }
    void configure(const std::vector<std::string>& data, const std::vector<std::string>& appends,
                   int64_t kv_heads, int64_t head_dim) {
        auto replacement = std::make_shared<Storage>(data, appends, kv_heads, head_dim);
        std::lock_guard<std::mutex> lock(api_mu);
        auto error = drain_locked();
        if (error) std::rethrow_exception(error);
        storage = std::move(replacement);
    }
    int submit(const Plan& input) {
        std::lock_guard<std::mutex> lock(api_mu);
        Prepared plan = prepare(input, storage);
        if (plan.tokens.empty()) return -1;
        int device;
        cuda_check(cudaGetDevice(&device));
        if (!readers) readers = std::make_unique<Executor>(2, stats);
        if (next_handle == std::numeric_limits<int>::max()) throw std::runtime_error("I/O handle limit reached");
        const int handle = next_handle++;
        auto job = std::make_shared<ReadJob>();
        job->plan = std::move(plan);
        requests.emplace(handle, job);
        try { readers->submit([this, job, device] { run(job, device); }); }
        catch (...) { requests.erase(handle); throw; }
        return handle;
    }
    std::shared_ptr<ReadJob> sync(const Plan& input) {
        std::shared_ptr<ReadJob> job;
        std::shared_ptr<Reader> reader;
        int handle, device;
        {
            std::lock_guard<std::mutex> lock(api_mu);
            auto plan = prepare(input, storage);
            if (plan.tokens.empty()) return nullptr;
            cuda_check(cudaGetDevice(&device));
            auto& context = synchronous_readers[device];
            if (!context) context = std::make_shared<Reader>(device, stats);
            reader = context;
            if (next_handle == std::numeric_limits<int>::max()) throw std::runtime_error("I/O handle limit reached");
            handle = next_handle++;
            job = std::make_shared<ReadJob>();
            job->plan = std::move(plan);
            requests.emplace(handle, job);
        }
        run(job, device, std::move(reader));
        { std::lock_guard<std::mutex> lock(api_mu); requests.erase(handle); }
        job->ready.get();
        return job;
    }
    std::shared_ptr<ReadJob> take(int handle) {
        std::shared_ptr<ReadJob> job;
        {
            std::lock_guard<std::mutex> lock(api_mu);
            auto it = requests.find(handle);
            if (it == requests.end()) throw std::runtime_error("Unknown or already-consumed I/O handle");
            job = std::move(it->second);
            requests.erase(it);
        }
        job->ready.get();
        return job;
    }
    void synchronize() {
        std::lock_guard<std::mutex> lock(api_mu);
        auto error = drain_locked();
        if (error) std::rethrow_exception(error);
    }
    void close() {
        std::lock_guard<std::mutex> lock(api_mu);
        auto error = drain_locked();
        readers.reset();
        synchronous_readers.clear();
        writer.reset();
        storage.reset();
        if (error) std::rethrow_exception(error);
    }
    bool append(const py::list& input) {
        std::shared_ptr<Storage> geometry;
        { std::lock_guard<std::mutex> lock(api_mu); geometry = storage; }
        if (!geometry) throw std::runtime_error("Storage not configured");
        const size_t block_bytes = geometry->block_bytes;
        struct Payload { int64_t disk, offset; torch::Tensor data; };
        std::vector<Payload> payloads;
        for (const auto& item : input) {
            auto row = item.cast<py::tuple>();
            if (row.size() != 4) throw std::runtime_error("Append request needs disk, offset, K, V");
            auto offset = row[1].cast<int64_t>();
            auto k = row[2].cast<torch::Tensor>(), v = row[3].cast<torch::Tensor>();
            if (offset < 0 || offset % block_bytes || offset > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(block_bytes) ||
                k.numel() != geometry->kv_heads * geometry->head_dim || v.numel() != k.numel() ||
                k.scalar_type() != torch::kBFloat16 || v.scalar_type() != torch::kBFloat16)
                throw std::runtime_error("Append needs BF16 KV matching storage geometry and an aligned record offset");
            auto data = torch::zeros({static_cast<int64_t>(block_bytes / 2)}, torch::dtype(torch::kBFloat16));
            data.narrow(0, 0, k.numel()).copy_(k.cpu().reshape({-1}));
            data.narrow(0, k.numel(), v.numel()).copy_(v.cpu().reshape({-1}));
            payloads.push_back({row[0].cast<int64_t>(), offset, std::move(data)});
        }
        if (payloads.empty()) return true;
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(api_mu);
        if (storage != geometry) throw std::runtime_error("Storage changed during append preparation");
        std::vector<AppendTask> tasks;
        for (auto& payload : payloads)
            tasks.push_back({&storage->file(payload.disk, 1), static_cast<size_t>(payload.offset), std::move(payload.data)});
        if (!writer) writer = std::make_unique<Executor>(1, stats);
        auto promise = std::make_shared<std::promise<void>>();
        auto ready = promise->get_future().share();
        auto target = storage;
        writer->submit([this, target, tasks = std::move(tasks), promise, block_bytes] {
            try {
                if (target->write_error) std::rethrow_exception(target->write_error);
                for (const auto& task : tasks) {
                    int fd = task.file->writer(stats);
                    size_t done = 0;
                    while (done < block_bytes) {
                        ssize_t n = pwrite(fd, static_cast<char*>(task.data.data_ptr()) + done, block_bytes - done, task.offset + done);
                        if (n < 0 && errno == EINTR) continue;
                        if (n <= 0) throw std::runtime_error("Failed or short append write");
                        done += n;
                    }
                }
                ++stats.write_jobs;
                promise->set_value();
            } catch (...) {
                target->write_error = std::current_exception();
                promise->set_exception(target->write_error);
            }
        });
        storage->last_write = std::move(ready);
        return true;
    }
    std::map<std::string, uint64_t> get_stats() const { return stats.snapshot(); }
};

py::tuple unpack(const std::shared_ptr<ReadJob>& job) {
    if (!job) return py::make_tuple(py::none(), py::none(), py::list());
    auto& buffer = job->buffer;
    // The worker has completed H2D. Track the consumer stream so its later
    // attention kernels finish before PyTorch recycles this output allocation.
    c10::cuda::CUDACachingAllocator::recordStream(buffer.storage().data_ptr(),
        c10::cuda::getCurrentCUDAStream(buffer.get_device()));
    auto& layout = *job->plan.storage;
    auto kv = buffer.narrow(1, 0, layout.payload_bytes / 2).view({buffer.size(0), 2, layout.kv_heads, layout.head_dim});
    return py::make_tuple(kv.select(1, 0).permute({1, 0, 2}).unsqueeze(0),
                         kv.select(1, 1).permute({1, 0, 2}).unsqueeze(0), job->plan.tokens);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto backend = std::make_shared<Backend>();
    m.doc() = "Octopus reusable io_uring/CUDA KV retrieval backend";
    m.def("configure_storage", [backend](const std::vector<std::string>& paths, const std::vector<std::string>& appends,
                                        int64_t kv_heads, int64_t head_dim) {
        backend->configure(paths, appends, kv_heads, head_dim);
    }, py::arg("paths"), py::arg("appends"), py::arg("kv_heads") = 8, py::arg("head_dim") = 128,
       py::call_guard<py::gil_scoped_release>());
    m.def("set_debug_logging", [backend](bool enabled, const std::string& path) { backend->set_debug(enabled, path); });
    m.def("sync_read_clusters", [backend](int64_t, const Plan& plan) {
        std::shared_ptr<ReadJob> job;
        { py::gil_scoped_release release; job = backend->sync(plan); }
        return unpack(job);
    });
    m.def("async_read_clusters", [backend](int64_t, const Plan& plan) { return backend->submit(plan); }, py::call_guard<py::gil_scoped_release>());
    m.def("wait_and_get", [backend](int handle) {
        std::shared_ptr<ReadJob> job;
        if (handle != -1) { py::gil_scoped_release release; job = backend->take(handle); }
        return unpack(job);
    });
    m.def("async_append_kv", [backend](const py::list& input) { return backend->append(input); });
    m.def("synchronize", [backend] { backend->synchronize(); }, py::call_guard<py::gil_scoped_release>());
    m.def("close", [backend] { backend->close(); }, py::call_guard<py::gil_scoped_release>());
    m.def("get_stats", [backend] { return backend->get_stats(); });
    py::module_::import("atexit").attr("register")(m.attr("close"));
}
