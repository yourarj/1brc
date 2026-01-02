use std::{
    arch::x86_64::*, // x86_64 SIMD intrinsics including AES-NI
    hash::{BuildHasher, Hasher},
};

#[derive(Default)]
pub(super) struct SimdHasher {
    state: __m128i, // 128-bit SSE register for hash state
    length: u64,    // Track input length for better mixing
}

impl SimdHasher {
    // Prime multiplier for final mixing - provides good avalanche effect
    const MULTIPLIER: u64 = 0x517cc1b727220a95;
    // Seed value for initialization
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
}

impl Hasher for SimdHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Track total bytes processed
        self.length = self.length.wrapping_add(bytes.len() as u64);

        unsafe {
            // Initialize state on first call (check if zero)
            if _mm_testz_si128(self.state, self.state) != 0 {
                // Set state with seed and length in two 64-bit halves
                self.state = _mm_set_epi64x(
                    Self::SEED as i64,  // High 64 bits
                    self.length as i64, // Low 64 bits
                );
            }

            let mut i = 0;

            // Process 16-byte chunks with AES-NI hardware acceleration
            while i + 16 <= bytes.len() {
                // Load 16 bytes unaligned (handles any address)
                let chunk = _mm_loadu_si128(bytes[i..].as_ptr() as *const __m128i);

                // Use AES encryption round for mixing
                // AES-NI instruction: one round of AES encryption
                // Exploits dedicated hardware for extremely fast mixing
                // Provides excellent avalanche and diffusion properties
                #[cfg(target_feature = "aes")]
                {
                    // _mm_aesenc_si128: state = ShiftRows(state) ⊕ SubBytes(state) ⊕ MixColumns(state) ⊕ chunk
                    // Single CPU instruction, ~3-7 cycles latency, throughput ~1 cycle
                    self.state = _mm_aesenc_si128(self.state, chunk);
                }

                // Fallback for CPUs without AES-NI
                #[cfg(not(target_feature = "aes"))]
                {
                    // XOR current state with input chunk
                    self.state = _mm_xor_si128(self.state, chunk);
                    // Shuffle 32-bit elements for mixing: [A,B,C,D] -> [B,C,D,A]
                    // Immediate value 0b10_11_00_01 specifies shuffle pattern
                    self.state = _mm_shuffle_epi32(self.state, 0b10_11_00_01);
                }

                i += 16;
            }

            // Process remaining 8-byte chunk if present
            if i + 8 <= bytes.len() {
                // Extract 8 bytes as u64
                let chunk = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
                // Pack into 128-bit vector with length as second element
                let chunk_vec = _mm_set_epi64x(chunk as i64, self.length as i64);

                #[cfg(target_feature = "aes")]
                {
                    // Mix with AES round
                    self.state = _mm_aesenc_si128(self.state, chunk_vec);
                }
                #[cfg(not(target_feature = "aes"))]
                {
                    // XOR fallback
                    self.state = _mm_xor_si128(self.state, chunk_vec);
                }

                i += 8;
            }

            // Process final bytes (< 8 remaining)
            if i < bytes.len() {
                // Create zero-padded 8-byte buffer
                let mut tail = [0u8; 8];
                // Copy remaining bytes into buffer
                tail[..bytes.len() - i].copy_from_slice(&bytes[i..]);
                // Convert to u64
                let tail_val = u64::from_ne_bytes(tail);
                // Pack into 128-bit vector
                let tail_vec = _mm_set_epi64x(tail_val as i64, 0);

                #[cfg(target_feature = "aes")]
                {
                    // Final mix with AES round
                    self.state = _mm_aesenc_si128(self.state, tail_vec);
                }
                #[cfg(not(target_feature = "aes"))]
                {
                    // Final XOR
                    self.state = _mm_xor_si128(self.state, tail_vec);
                }
            }
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        unsafe {
            // Extract low 64 bits from 128-bit state
            let low = _mm_extract_epi64(self.state, 0) as u64;
            // Extract high 64 bits from 128-bit state
            let high = _mm_extract_epi64(self.state, 1) as u64;

            // Final mixing: multiply-xor to fold 128 bits into 64 bits
            // Provides final avalanche for uniformly distributed hash
            low.wrapping_mul(Self::MULTIPLIER) ^ high
        }
    }
}

#[derive(Default, Clone, Copy)]
pub(super) struct SimdBuildHasher;

impl BuildHasher for SimdBuildHasher {
    type Hasher = SimdHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        SimdHasher {
            state: unsafe { _mm_setzero_si128() }, // Zero-initialize SSE register
            length: 0,
        }
    }
}
