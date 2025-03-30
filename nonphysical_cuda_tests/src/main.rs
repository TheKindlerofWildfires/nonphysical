use std::time::SystemTime;

use nonphysical_cuda::cuda::runtime::Runtime;
use nonphysical_ptx::crypt::hash::cuda::Hasher;

/*
fn sort_test() {
    let mut data = (0..1024 * 1024)
        .map(|i| (F32::isize(i).sin() * F32::usize(100)).as_usize())
        .collect::<Vec<_>>();

    let now = SystemTime::now();
    data.sort();

    dbg!(now.elapsed());
    Runtime::init(0, "nonphysical_ptx.ptx");
    let data1 = (0..8 * 8).collect::<Vec<_>>();

    //let data = vec![9,7,5,3,8,4,6,5];
    let data1 = data1
        .into_iter()
        .rev()
        .map(|i| F32::isize(i))
        .collect::<Vec<_>>();
    let out = CudaMergeSort::merge_1024(&data1);
    dbg!(out.len(), &out);
}*/
/*
    Runtime::init(0, "nonphysical_ptx.ptx");
    let ncst = 2048;
    let mut data = (0..ncst*128).map(|i| ComplexScaler::new(F32::usize(i%ncst), F32::ZERO)).collect::<Vec<_>>();
    let cst = CycloStationaryTransformCuda::new(ncst);
    let now = SystemTime::now();

    let _ = cst.fam(&mut data);
    let _ = dbg!(now.elapsed());
*/
pub fn main() {
    //dbg!(out);
    Runtime::init(0, "nonphysical_ptx.ptx");
    let hex = "ee26b0dd4af7e749aa1a8ee3c10ae9923f618980772e473f8819a5d4940e0db27ac185f8a0e1d5f84f88bc887fd67b143732c304cc5fa9ad8e6f57f50028a8ff";
    let mut bytes = Vec::with_capacity(hex.len() / 2);

    for i in (0..hex.len()).step_by(2) {
        // Get two characters from the hex string
        let byte_str = &hex[i..i + 2];

        // Parse the byte and push it to the result vector
        match u8::from_str_radix(byte_str, 16) {
            Ok(byte) => bytes.push(byte),
            Err(_) => panic!(),
        }
    }
    let hasher = Hasher::new(&bytes);
    let now = SystemTime::now();
    dbg!("Starting crack!");

    let hit = hasher.crack("sha512", "alphanumeric");
    dbg!(hit);
    let _ = dbg!(now.elapsed());
    //dbg!(out.rows,out.cols,&out.data);
}
