use std::{hash::BuildHasher, simd::Simd};

#[derive(Default)]
pub(super) struct SimdHasher {
    state: u64,
}

impl std::hash::Hasher for SimdHasher {
    fn write(&mut self, bytes: &[u8]) {
        // We use 16-byte chunks as baseline (128-bit SIMD) but detect AVX2 for 32-byte chunks
        #[cfg(target_feature = "avx2")]
        const CHUNK_SIZE: usize = 32; // 256-bit AVX2
        #[cfg(not(target_feature = "avx2"))]
        const CHUNK_SIZE: usize = 16; // 128-bit SSE

        let chunks = bytes.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Load entire chunk into SIMD register at once
            let simd_vec = Simd::<u8, CHUNK_SIZE>::from_slice(chunk);
            let bytes: [u8; CHUNK_SIZE] = simd_vec.to_array();

            // Process in 8-byte blocks - optimal for u64 operations
            for i in (0..CHUNK_SIZE).step_by(8) {
                let block = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
                // Mix block with hash state using XOR and prime multiplication
                self.state = (block ^ self.state).wrapping_mul(0x517cc1b727220a95);
            }
        }

        // Process remaining bytes in 8-byte blocks where possible
        let mut i = 0;
        while i < remainder.len() {
            if i + 8 <= remainder.len() {
                let block = u64::from_ne_bytes(remainder[i..i + 8].try_into().unwrap());
                self.state = (block ^ self.state).wrapping_mul(0x517cc1b727220a95);
                i += 8;
            } else {
                // Final single-byte processing
                self.state = (remainder[i] as u64 ^ self.state).wrapping_mul(0x517cc1b727220a95);
                i += 1;
            }
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

#[derive(Default)]
pub(super) struct SimdBuildHasher;

impl BuildHasher for SimdBuildHasher {
    type Hasher = SimdHasher;

    fn build_hasher(&self) -> Self::Hasher {
        SimdHasher::default()
    }
}
