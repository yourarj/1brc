#![feature(portable_simd)]  // Enable experimental SIMD support for portable vectorization

mod simd_hasher;

use std::{
    collections::BTreeMap,  // For sorted output of station names
    fs::File,
    os::fd::AsRawFd,  // Unix file descriptor access for mmap
    ptr,
    simd::{u8x32, Simd, SimdPartialEq},  // AVX2 256-bit SIMD vectors for parallel byte operations
    sync::Arc,  // Thread-safe reference counting for sharing data across threads
    thread,
};

use libc::mmap;
use simd_hasher::SimdBuildHasher;

// Type alias using our custom hasher for all HashMaps in this file
type HashMap<K, V> = std::collections::HashMap<K, V, SimdBuildHasher>;

// AVX2 processes 32 bytes (256 bits) per SIMD instruction
const SIMD_WIDTH: usize = 32;

fn main() {
    // Open the input file containing station;temperature pairs
    let f = File::open("../measurements.txt").unwrap();
    
    // Memory-map the file for zero-copy access - avoids buffering overhead
    let map = memmap(&f);

    // Determine optimal parallelism: use all available CPU cores
    let n_threads = thread::available_parallelism().unwrap().get();
    
    // Divide file into approximately equal chunks (one per thread)
    let chunk_size = map.len() / n_threads;
    
    // Split at newline boundaries to avoid splitting lines across chunks
    let chunks = split_at_newlines(map, chunk_size);
    
    // Wrap in Arc for safe sharing across threads without copying
    let chunks = Arc::new(chunks);

    // Spawn one thread per chunk for parallel processing
    let handles: Vec<_> = (0..chunks.len())
        .map(|i| {
            let chunks = Arc::clone(&chunks);  // Clone Arc pointer, not data
            thread::spawn(move || process_chunk(chunks[i]))
        })
        .collect();

    // Create final hashmap with our AES-NI accelerated hasher
    let mut stats = HashMap::with_hasher(SimdBuildHasher);
    
    // Wait for all threads and merge their results
    for handle in handles {
        let chunk_stats = handle.join().unwrap();
        
        // Merge each thread's statistics into global stats
        for (station, (min, sum, count, max)) in chunk_stats {
            stats
                .entry(station)
                .and_modify(|e: &mut (i16, i64, usize, i16)| {
                    // Update min/max using branchless min/max operations
                    e.0 = e.0.min(min);  // Minimum temperature
                    e.1 += sum;          // Sum for calculating mean
                    e.2 += count;        // Count of measurements
                    e.3 = e.3.max(max);  // Maximum temperature
                })
                .or_insert((min, sum, count, max));  // First time seeing this station
        }
    }

    print_results(&stats);
}

/// Split buffer at newline boundaries for parallel processing
/// Ensures no line is split across chunks
fn split_at_newlines(data: &[u8], approximate_chunk_size: usize) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < data.len() {
        // Calculate target end position
        let end = (start + approximate_chunk_size).min(data.len());
        
        let actual_end = if end < data.len() {
            // Not at file end - find next newline after chunk boundary
            // This ensures we don't split a line across chunks
            end + memchr(b'\n', &data[end..]).unwrap_or(data.len() - end) + 1
        } else {
            // At file end - take everything remaining
            data.len()
        };

        // Store slice reference (zero-copy)
        chunks.push(&data[start..actual_end]);
        start = actual_end;
    }

    chunks
}

/// SIMD-accelerated byte search using AVX2 - 32 bytes per iteration
/// Replaces sequential byte-by-byte search with parallel comparison
#[inline]
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    // Broadcast needle byte to all 32 lanes of SIMD vector
    let needle_simd = Simd::splat(needle);
    let mut pos = 0;

    // Main SIMD loop: process 32 bytes per iteration using AVX2
    while pos + SIMD_WIDTH <= haystack.len() {
        // Load 32 bytes from haystack into SIMD register
        let chunk = u8x32::from_slice(&haystack[pos..pos + SIMD_WIDTH]);
        
        // Parallel comparison: compare all 32 bytes simultaneously
        // Creates a bitmask where matching positions are set
        let mask = chunk.simd_eq(needle_simd);

        // Check if any lane matched
        if mask.any() {
            // Find first matching position by testing each lane
            // Modern CPUs can optimize this with BSF (bit scan forward)
            for i in 0..SIMD_WIDTH {
                if mask.test(i) {
                    return Some(pos + i);
                }
            }
        }
        pos += SIMD_WIDTH;
    }

    // Scalar fallback for final bytes (< 32 remaining)
    // Uses standard iterator - compiler may auto-vectorize
    haystack[pos..].iter().position(|&b| b == needle).map(|p| pos + p)
}

/// Process a single chunk of data in one thread
/// Returns a local HashMap to be merged later
fn process_chunk(data: &[u8]) -> HashMap<Vec<u8>, (i16, i64, usize, i16)> {
    // Each thread gets its own HashMap - no synchronization needed
    let mut stats = HashMap::with_hasher(SimdBuildHasher);
    let mut pos = 0;

    // Parse line-by-line using SIMD newline detection
    while pos < data.len() {
        // Find end of current line using SIMD memchr
        let line_end = match memchr(b'\n', &data[pos..]) {
            Some(offset) => pos + offset,
            None => break,  // No more complete lines
        };

        // Process non-empty lines only
        if line_end > pos {
            process_line(&data[pos..line_end], &mut stats);
        }

        pos = line_end + 1;  // Skip the newline character
    }

    stats
}

/// Process single line - optimized for minimal branching
/// Format: "station_name;temperature"
#[inline(always)]  // Force inline for hot path optimization
fn process_line(line: &[u8], stats: &mut HashMap<Vec<u8>, (i16, i64, usize, i16)>) {
    // Find semicolon delimiter using SIMD-accelerated search
    let semicolon_pos = memchr(b';', line).unwrap();
    
    // Split line: everything before semicolon is station name
    let station = &line[..semicolon_pos];
    // Everything after semicolon is temperature string
    let temp_bytes = &line[semicolon_pos + 1..];
    
    // Parse temperature to i16 (value * 10, e.g., 12.3 -> 123)
    let temperature = parse_temp_fast(temp_bytes);

    // Update or insert statistics for this station
    stats
        .entry(station.to_vec())  // Convert slice to owned Vec for key
        .and_modify(|e| {
            // Station exists - update min/max/sum/count
            e.0 = e.0.min(temperature);  // Branchless min
            e.1 += i64::from(temperature);  // Accumulate for mean
            e.2 += 1;  // Increment count
            e.3 = e.3.max(temperature);  // Branchless max
        })
        .or_insert((temperature, i64::from(temperature), 1, temperature));
        // First occurrence - initialize all stats
}

/// Branchless temperature parsing optimized for known format
/// Temperatures are always -99.9 to 99.9 with one decimal place
/// Returns value * 10 (e.g., -12.3 returns -123)
#[inline(always)]
fn parse_temp_fast(temp: &[u8]) -> i16 {
    let len = temp.len();
    let mut idx = 0;
    let mut sign = 1i16;

    // Check for negative sign - branch predictor friendly (biased branch)
    if temp[0] == b'-' {
        sign = -1;
        idx = 1;
    }

    let mut value = 0i16;
    
    // Parse all digits, skip decimal point
    // Loop unrolls well since max 4 iterations
    while idx < len {
        let byte = temp[idx];
        // Skip decimal point character, process digits
        if byte != b'.' {
            // Multiply-add: value = value * 10 + digit
            // Uses hardware multiply on modern CPUs
            value = value * 10 + (byte - b'0') as i16;
        }
        idx += 1;
    }

    // Apply sign: branchless on many architectures via CMOV
    value * sign
}

/// Print results in the required format
fn print_results(stats: &HashMap<Vec<u8>, (i16, i64, usize, i16)>) {
    print!("{{");
    
    // Convert to BTreeMap for sorted output by station name
    let stats = BTreeMap::from_iter(
        stats
            .iter()
            .map(|(k, v)| (
                // SAFETY: Station names are valid UTF-8 in challenge
                unsafe { String::from_utf8_unchecked(k.to_vec()) }, 
                v
            )),
    );
    
    let mut stats = stats.into_iter().peekable();

    // Print each station's statistics
    while let Some((station, (min, sum, count, max))) = stats.next() {
        // Format: name=min/mean/max (divide by 10 to restore decimal)
        print!(
            "{station}={:.1}/{:.1}/{:.1}",
            (*min as f64) / 10.,  // Min temperature
            (*sum as f64) / 10. / (*count as f64),  // Mean temperature
            (*max as f64) / 10.  // Max temperature
        );

        // Add comma separator between stations (not after last)
        if stats.peek().is_some() {
            print!(", ")
        }
    }
    println!("}}");
}

/// Memory-map the file for zero-copy access
/// Uses Unix mmap for direct page cache access
fn memmap(f: &File) -> &'_ [u8] {
    let len = f.metadata().unwrap().len();
    
    unsafe {
        // Map file into process address space
        let ptr = mmap(
            ptr::null_mut(),  // Let kernel choose address
            len as libc::size_t,  // Map entire file
            libc::PROT_READ,  // Read-only access
            libc::MAP_SHARED | libc::MAP_POPULATE,  // Shared mapping + prefault pages
            f.as_raw_fd(),  // File descriptor
            0,  // Offset: start from beginning
        );
        
        // Check for mapping failure
        if ptr == libc::MAP_FAILED {
            panic!("{:?}", std::io::Error::last_os_error());
        }
        
        // Hint to kernel about access pattern for prefetching optimization
        // MADV_SEQUENTIAL: read sequentially, aggressive readahead
        // MADV_WILLNEED: start prefetching now
        if libc::madvise(
            ptr, 
            len as libc::size_t, 
            libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED
        ) != 0 {
            panic!("{:?}", std::io::Error::last_os_error());
        }
        
        // Create slice view over mapped memory
        core::slice::from_raw_parts(ptr as *const u8, len as usize)
    }
}
