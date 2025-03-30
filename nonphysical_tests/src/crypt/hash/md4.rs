#[cfg(test)]
mod md4_tests {
    use nonphysical_core::crypt::hash::md4::MD4;
    #[test]
    pub fn test() {
        let body = "tes";
        let input = body.chars().map(|c| c as u8).collect::<Vec<_>>();
        let mut hasher = MD4::new();
        let digest = hasher.digest(&input);
        let mut buffer = String::new();
        digest.iter().for_each(|byte| {
            buffer.push_str(&format!("{:02x}", byte));
        });
        dbg!(buffer);
    }
}
