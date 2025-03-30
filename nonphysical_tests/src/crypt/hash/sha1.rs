#[cfg(test)]
mod sha1_tests {
    use nonphysical_core::crypt::hash::sha1::SHA1;
    #[test]
    pub fn test() {
        let body = "sha1";
        let input = body.chars().map(|c| c as u8).collect::<Vec<_>>();
        let mut hasher = SHA1::new();
        let digest = hasher.digest(&input);
        let mut buffer = String::new();
        digest.iter().for_each(|byte| {
            buffer.push_str(&format!("{:02x}", byte));
        });
        dbg!(buffer);
    }
}
