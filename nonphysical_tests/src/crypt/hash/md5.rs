#[cfg(test)]
mod md5_tests {
    use nonphysical_core::crypt::hash::md5::MD5;
    #[test]
    pub fn test() {
        let body = "test";
        let input = body.chars().map(|c| c as u8).collect::<Vec<_>>();
        let mut hasher = MD5::new();
        let digest = hasher.digest(&input);
        let mut buffer = String::new();
        digest.iter().for_each(|byte| {
            buffer.push_str(&format!("{:02x}", byte));
        });
        dbg!(buffer);
    }
}
