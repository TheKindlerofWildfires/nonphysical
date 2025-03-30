#[cfg(test)]
mod sha2_tests {
    use nonphysical_core::crypt::hash::sha2::SHA256;
    use nonphysical_core::crypt::hash::sha2::SHA512;
    #[test]
    pub fn test_256() {
        let body = "test";
        let input = body.chars().map(|c| c as u8).collect::<Vec<_>>();
        let mut hasher = SHA256::new();
        let digest = hasher.digest(&input);
        let mut buffer = String::new();
        digest.iter().for_each(|byte| {
            buffer.push_str(&format!("{:02x}", byte));
        });
        dbg!(buffer);
    }
    #[test]
    pub fn test_512() {
        let body = "test";
        let input = body.chars().map(|c| c as u8).collect::<Vec<_>>();
        let mut hasher = SHA512::new();
        let digest = hasher.digest(&input);
        let mut buffer = String::new();
        digest.iter().for_each(|byte| {
            buffer.push_str(&format!("{:02x}", byte));
        });
        dbg!(buffer);
    }
}
