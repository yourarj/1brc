#![feature(portable_simd)]

mod simd_hasher;
mod simd_newline;

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    os::fd::AsRawFd,
    ptr,
};

use libc::mmap;
use simd_hasher::SimdBuildHasher;
use simd_newline::{find_next_newline, find_next_semicolon};

fn main() {
    let f = fs::File::open("../measurements.txt").unwrap();
    let map = memmap(&f);

    let mut stats =
        HashMap::<Vec<u8>, (i16, i64, usize, i16), SimdBuildHasher>::with_hasher(SimdBuildHasher);

    let mut start = 0;
    while let Some(pos) = find_next_newline(&map[start..]) {
        let end = start + pos;

        let line = &map[start..end];
        if line.is_empty() {
            break;
        }

        let semicolon_pos = match find_next_semicolon(line) {
            Some(pos) => pos,
            None => continue,
        };
        let temperature = parse_temp(&line[semicolon_pos + 1..]);
        let station = &line[..semicolon_pos];

        let stats = match stats.get_mut(station) {
            Some(stats) => stats,
            None => stats
                .entry(station.to_vec())
                .or_insert((i16::MAX, 0, 0, i16::MIN)),
        };

        stats.0 = stats.0.min(temperature);
        stats.1 += i64::from(temperature);
        stats.2 += 1;
        stats.3 = stats.3.max(temperature);

        start = end + 1;
    }

    print!("{{");
    let stats = BTreeMap::from_iter(
        stats
            .iter()
            .map(|(k, v)| (unsafe { String::from_utf8_unchecked(k.to_vec()) }, v)),
    );
    let mut stats = stats.into_iter().peekable();

    while let Some((station, (min, sum, count, max))) = stats.next() {
        print!(
            "{station}={:.1}/{:.1}/{:.1}",
            (*min as f64) / 10.,
            (*sum as f64) / 10. / (*count as f64),
            (*max as f64) / 10.
        );

        if stats.peek().is_some() {
            print!(", ")
        }
    }
    print!("}}");
}

/**
 * parsing logic
 */
fn parse_temp(temp: &[u8]) -> i16 {
    let mut t: i16 = 0;
    let mut mul = 1;

    for &d in temp.iter().rev() {
        match d {
            b'.' => {
                continue;
            }
            b'-' => {
                t = -t;
                break;
            }
            _ => {
                t += i16::from(d - b'0') * mul;
                mul *= 10;
            }
        }
    }
    t
}

fn memmap(f: &File) -> &'_ [u8] {
    let len = f.metadata().unwrap().len();
    unsafe {
        let ptr = mmap(
            ptr::null_mut(),
            len as libc::size_t,
            libc::PROT_READ,
            libc::MAP_SHARED,
            f.as_raw_fd(),
            0,
        );
        if ptr == libc::MAP_FAILED {
            panic!("{:?}", std::io::Error::last_os_error());
        } else {
            if libc::madvise(ptr, len as libc::size_t, libc::MADV_SEQUENTIAL) != 0 {
                panic!("{:?}", std::io::Error::last_os_error());
            }
            core::slice::from_raw_parts(ptr as *const u8, len as usize)
        }
    }
}
