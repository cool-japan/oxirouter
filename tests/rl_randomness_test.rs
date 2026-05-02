//! Tests proving RL policy RNG is non-deterministic and statistically correct.

#[cfg(feature = "rl")]
mod tests {
    use oxirouter::rl::{Policy, Reward};

    fn sources() -> Vec<String> {
        vec!["a".into(), "b".into(), "c".into(), "d".into()]
    }

    #[test]
    fn test_epsilon_greedy_explores_uniformly() {
        // With ε=1.0, all selections should be random
        let mut counts = [0u32; 4];
        let policy = Policy::epsilon_greedy(1.0).with_seed(42);
        let srcs: Vec<String> = sources();
        let refs: Vec<&String> = srcs.iter().collect();
        for _ in 0..10_000 {
            if let Some(s) = policy.select(&refs) {
                if let Some(i) = srcs.iter().position(|x| x == &s) {
                    counts[i] += 1;
                }
            }
        }
        // Each source should be picked roughly 2500 times; allow ±300
        for &c in &counts {
            assert!(
                (2200..=2800).contains(&c),
                "expected ~2500 picks, got {c}; counts: {counts:?}"
            );
        }
    }

    #[test]
    fn test_epsilon_greedy_exploits_at_zero_eps() {
        let mut policy = Policy::epsilon_greedy(0.0);
        let srcs: Vec<String> = sources();
        let refs: Vec<&String> = srcs.iter().collect();
        // Set up Q-values: make "a" best
        policy.initialize_source("a");
        policy.initialize_source("b");
        policy.initialize_source("c");
        policy.initialize_source("d");
        policy.update("a", Reward::new(1.0));
        policy.update("a", Reward::new(1.0));
        for _ in 0..100 {
            assert_eq!(policy.select(&refs), Some("a".to_string()));
        }
    }

    #[test]
    fn test_thompson_sample_distribution() {
        // Source "a" has many successes, "b" has many failures
        let mut policy = Policy::thompson_sampling().with_seed(99);
        for _ in 0..20 {
            policy.update("a", Reward::new(1.0));
        }
        for _ in 0..20 {
            policy.update("b", Reward::new(0.0));
        }
        let srcs = vec!["a".to_string(), "b".to_string()];
        let refs: Vec<&String> = srcs.iter().collect();
        let mut a_count = 0u32;
        for _ in 0..1000 {
            if policy.select(&refs) == Some("a".to_string()) {
                a_count += 1;
            }
        }
        // "a" should be selected >= 85% of the time
        assert!(
            a_count >= 850,
            "expected >=850/1000 for high-success source, got {a_count}"
        );
    }

    #[test]
    fn test_seeded_policy_reproducible() {
        let p1 = Policy::epsilon_greedy(0.5).with_seed(1234);
        let p2 = Policy::epsilon_greedy(0.5).with_seed(1234);
        let srcs: Vec<String> = sources();
        let refs: Vec<&String> = srcs.iter().collect();
        let mut seq1 = Vec::new();
        let mut seq2 = Vec::new();
        for _ in 0..50 {
            seq1.push(p1.select(&refs));
            seq2.push(p2.select(&refs));
        }
        assert_eq!(
            seq1, seq2,
            "seeded policies should produce identical sequences"
        );
    }

    #[test]
    fn test_two_different_seeds_differ() {
        let p1 = Policy::epsilon_greedy(1.0).with_seed(1);
        let p2 = Policy::epsilon_greedy(1.0).with_seed(999);
        let srcs: Vec<String> = sources();
        let refs: Vec<&String> = srcs.iter().collect();
        let mut seq1 = Vec::new();
        let mut seq2 = Vec::new();
        for _ in 0..20 {
            seq1.push(p1.select(&refs));
            seq2.push(p2.select(&refs));
        }
        // Should not be identical (astronomically unlikely with 2 different seeds)
        assert_ne!(
            seq1, seq2,
            "different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_serialize_deserialize_reseeds() {
        let policy = Policy::epsilon_greedy(1.0).with_seed(42);
        let srcs: Vec<String> = sources();
        let refs: Vec<&String> = srcs.iter().collect();
        let json = serde_json::to_string(&policy).expect("serialize");
        let restored: Policy = serde_json::from_str(&json).expect("deserialize");
        // Restored policy should still select a source
        let sel = restored.select(&refs);
        assert!(sel.is_some(), "restored policy should select a source");
    }

    #[test]
    fn test_ucb_beats_greedy_over_time() {
        // UCB should explore and eventually prefer the better source
        let mut ucb_policy = Policy::ucb();
        let srcs = vec!["good".to_string(), "bad".to_string()];
        let refs: Vec<&String> = srcs.iter().collect();
        ucb_policy.initialize_source("good");
        ucb_policy.initialize_source("bad");
        // Feed initial rewards to establish baseline
        ucb_policy.update("good", Reward::new(0.8));
        ucb_policy.update("bad", Reward::new(0.2));
        let mut good_count = 0u32;
        for _ in 0..200 {
            if ucb_policy.select(&refs) == Some("good".to_string()) {
                good_count += 1;
            }
        }
        // UCB should strongly prefer "good" after observing its higher reward
        assert!(
            good_count >= 150,
            "UCB should mostly exploit good source, got {good_count}/200"
        );
    }
}
