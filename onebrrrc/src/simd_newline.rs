use libc::memchr;

#[inline]
/// Find the next occurrence of a byte in a slice using memchr.
pub fn find_next_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    unsafe {
        let ptr = memchr(haystack.as_ptr() as *const libc::c_void, needle as i32, haystack.len() as libc::size_t);
        if ptr.is_null() {
            None
        } else {
            Some((ptr as usize) - (haystack.as_ptr() as usize))
        }
    }
}

/// Find the next newline character using memchr.
pub fn find_next_newline(haystack: &[u8]) -> Option<usize> {
    find_next_byte(haystack, b'\n')
}

/// Find the next semicolon character using memchr.
pub fn find_next_semicolon(haystack: &[u8]) -> Option<usize> {
    find_next_byte(haystack, b';')
}