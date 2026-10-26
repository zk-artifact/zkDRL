use super::*;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn sha(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

// Frozen with Python hashlib, independent of CounterRng and sha2.
#[test]
fn known_keys_blocks_and_little_endian_words() {
    {
        let seed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let challenge = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut rng = CounterRng::new(1, seed, challenge, sha);
        assert_eq!(hex(&rng.key), "77b5e977b322ab6eb46aede05902cb267946924466fa4facf35bf767ab226e50");
        let expected = [0x41b492f5u32, 0xc8f0d132u32, 0xdc9ef33bu32, 0x21fc97d4u32, 0x6f2842aau32, 0xfd26d102u32, 0xd831224fu32, 0x91680c83u32, 0x5fff5eb3u32, 0x762ead74u32, 0xf7e02d18u32, 0x2ac759a6u32, 0xb5a384b8u32, 0x2ccde03au32, 0x4e46a161u32, 0x1efe6bffu32, 0x42591753u32, 0x9c6c88a1u32, 0x9823175au32, 0x173fec95u32, 0xba5db45du32, 0x3807f6fcu32, 0x88352a48u32, 0xd5cbfc7bu32];
        let blocks = ["f592b44132d1f0c83bf39edcd497fc21aa42286f02d126fd4f2231d8830c6891", "b35eff5f74ad2e76182de0f7a659c72ab884a3b53ae0cd2c61a1464eff6bfe1e", "53175942a1886c9c5a17239895ec3f175db45dbafcf60738482a35887bfccbd5"];
        assert_eq!(rng.position(), 0);
        for (i, word) in expected.into_iter().enumerate() {
            assert_eq!(rng.next_u32(), word);
            assert_eq!(rng.position(), i as u64 + 1);
            assert_eq!(hex(&rng.block), blocks[i / 8]);
        }
    }
    {
        let seed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let challenge = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut rng = CounterRng::new(2, seed, challenge, sha);
        assert_eq!(hex(&rng.key), "8dab15bd59e9f8efff9de485144fd1c8a7e95038a51604cbd3857708bb49a10d");
        let expected = [0x71775e4cu32, 0x5f356d0eu32, 0x409b8be5u32, 0xcf94fa2du32, 0xa52007cfu32, 0x99d1cdd8u32, 0xe2a5aca1u32, 0x0bbb126bu32, 0x7e31795fu32, 0xb75ed4fdu32, 0xacf6aa9fu32, 0x8f3929e7u32, 0x5da69a99u32, 0x2e31f006u32, 0x0576fe80u32, 0x0954ececu32, 0x429cbddfu32, 0x41b319bdu32, 0x895cd47bu32, 0x59491608u32, 0x9c85f4aeu32, 0x15d925c4u32, 0x353b6d69u32, 0x0e4a5a69u32];
        let blocks = ["4c5e77710e6d355fe58b9b402dfa94cfcf0720a5d8cdd199a1aca5e26b12bb0b", "5f79317efdd45eb79faaf6ace729398f999aa65d06f0312e80fe7605ecec5409", "dfbd9c42bd19b3417bd45c8908164959aef4859cc425d915696d3b35695a4a0e"];
        assert_eq!(rng.position(), 0);
        for (i, word) in expected.into_iter().enumerate() {
            assert_eq!(rng.next_u32(), word);
            assert_eq!(rng.position(), i as u64 + 1);
            assert_eq!(hex(&rng.block), blocks[i / 8]);
        }
    }
    {
        let seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
        let challenge = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63];
        let mut rng = CounterRng::new(1, seed, challenge, sha);
        assert_eq!(hex(&rng.key), "1b38acec40832e496f963110caa1aa3ec4fc994a9acc4974e8d99f251e72154a");
        let expected = [0x343b5a23u32, 0x860149c8u32, 0xc65c0113u32, 0x22ec6f15u32, 0xe84ded92u32, 0xfc47e852u32, 0x07516541u32, 0x37aa63b5u32, 0xb871779cu32, 0x671395fcu32, 0xc440ab08u32, 0x51fa227fu32, 0xf81fc637u32, 0xab2146d9u32, 0xb4416874u32, 0x940d8d63u32, 0x83f2bf98u32, 0x3e5769bdu32, 0xaf336c8cu32, 0x3eeebbc4u32, 0x10d0fb7au32, 0x64d19e84u32, 0x00423c03u32, 0xdf64258du32];
        let blocks = ["235a3b34c849018613015cc6156fec2292ed4de852e847fc41655107b563aa37", "9c7771b8fc95136708ab40c47f22fa5137c61ff8d94621ab746841b4638d0d94", "98bff283bd69573e8c6c33afc4bbee3e7afbd010849ed164033c42008d2564df"];
        assert_eq!(rng.position(), 0);
        for (i, word) in expected.into_iter().enumerate() {
            assert_eq!(rng.next_u32(), word);
            assert_eq!(rng.position(), i as u64 + 1);
            assert_eq!(hex(&rng.block), blocks[i / 8]);
        }
    }
    {
        let seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
        let challenge = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63];
        let mut rng = CounterRng::new(2, seed, challenge, sha);
        assert_eq!(hex(&rng.key), "a6904ffecf75b18138c307b964e75600d1cafd8a829edac4dd246020dbce1796");
        let expected = [0x50510c52u32, 0xf1480643u32, 0x27265456u32, 0x83fb4fdbu32, 0x13919360u32, 0xae3a1242u32, 0x3063138eu32, 0xb51179e3u32, 0x7e850b4cu32, 0x65b9c334u32, 0x1b7f4ae3u32, 0xcc2e85b8u32, 0xedd5152au32, 0xd86580f4u32, 0x18ce547fu32, 0x49fb894bu32, 0xf415201au32, 0x4aa88d3fu32, 0x9d227b80u32, 0xc1e00aa3u32, 0xa62fa7e6u32, 0xdaaceb57u32, 0xae3ee47fu32, 0x71243f09u32];
        let blocks = ["520c5150430648f156542627db4ffb836093911342123aae8e136330e37911b5", "4c0b857e34c3b965e34a7f1bb8852ecc2a15d5edf48065d87f54ce184b89fb49", "1a2015f43f8da84a807b229da30ae0c1e6a72fa657ebacda7fe43eae093f2471"];
        assert_eq!(rng.position(), 0);
        for (i, word) in expected.into_iter().enumerate() {
            assert_eq!(rng.next_u32(), word);
            assert_eq!(rng.position(), i as u64 + 1);
            assert_eq!(hex(&rng.block), blocks[i / 8]);
        }
    }
}

#[test]
fn repeatability_and_domain_separation() {
    let mut a = CounterRng::new(DQN, [7; 32], [9; 32], sha);
    let mut repeat = CounterRng::new(DQN, [7; 32], [9; 32], sha);
    let mut challenge = CounterRng::new(DQN, [7; 32], [10; 32], sha);
    let mut seed = CounterRng::new(DQN, [8; 32], [9; 32], sha);
    let mut algorithm = CounterRng::new(A2C, [7; 32], [9; 32], sha);
    let take = |r: &mut CounterRng| (0..100).map(|_| r.next_u32()).collect::<Vec<_>>();
    let words = take(&mut a);
    assert_eq!(words, take(&mut repeat));
    assert_ne!(words, take(&mut challenge));
    assert_ne!(words, take(&mut seed));
    assert_ne!(words, take(&mut algorithm));
}

thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) }; }
fn counted_sha(bytes: &[u8]) -> [u8; 32] {
    CALLS.with(|c| c.set(c.get() + 1));
    sha(bytes)
}

#[test]
fn cache_block_boundaries_and_checked_cursor_exhaustion() {
    CALLS.with(|c| c.set(0));
    let mut rng = CounterRng::new(DQN, [0; 32], [0; 32], counted_sha);
    assert_eq!(CALLS.with(Cell::get), 1);
    for i in 0..17 {
        rng.next_u32();
        assert_eq!(CALLS.with(Cell::get), 2 + i / 8);
    }
    // Test-only private mutation; the public API has no cursor/cache input.
    rng.position = u64::MAX - 1;
    let mut preimage = BLOCK_DOMAIN.to_vec();
    preimage.extend_from_slice(&rng.key);
    preimage.extend_from_slice(&((u64::MAX - 1) / 8).to_le_bytes());
    let block = sha(&preimage);
    assert_eq!(rng.next_u32(), u32::from_le_bytes(block[24..28].try_into().unwrap()));
    assert_eq!(rng.position(), u64::MAX);
    let calls = CALLS.with(Cell::get);
    assert!(catch_unwind(AssertUnwindSafe(|| rng.next_u32())).is_err());
    assert_eq!(rng.position(), u64::MAX);
    assert_eq!(CALLS.with(Cell::get), calls);
    for invalid in [0, 3, u32::MAX] {
        assert!(catch_unwind(|| CounterRng::new(invalid, [0; 32], [0; 32], sha)).is_err());
    }
}
