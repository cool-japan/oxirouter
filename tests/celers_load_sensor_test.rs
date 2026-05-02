//! Tests for the Celers-backed load sensor.

#[cfg(feature = "load")]
mod load_sensor_tests {
    use celers_core::{BrokerStats, PoolStats, WorkerStats};
    use oxirouter::{CelersLoadSensor, context::sensor::LoadSensor};

    fn make_stats(
        total: u64,
        active: u32,
        succeeded: u64,
        failed: u64,
        loadavg: Option<[f64; 3]>,
        pool: Option<PoolStats>,
        broker: Option<BrokerStats>,
    ) -> WorkerStats {
        WorkerStats {
            total_tasks: total,
            active_tasks: active,
            succeeded,
            failed,
            retried: 0,
            uptime: 3600.0,
            loadavg,
            memory_usage: None,
            pool,
            broker,
            clock: None,
        }
    }

    #[test]
    fn active_tasks_maps_to_pending_tasks() {
        let sensor = CelersLoadSensor::new(|| make_stats(100, 7, 90, 3, None, None, None));
        let ctx = sensor.sense().expect("sensor must return Some");
        assert_eq!(ctx.pending_tasks, 7);
    }

    #[test]
    fn loadavg_maps_to_global_load() {
        let sensor = CelersLoadSensor::new(|| {
            make_stats(
                100,
                5,
                95,
                0,
                Some([0.5, 0.4, 0.3]), // 1-min load = 0.5 with default concurrency 1
                None,
                None,
            )
        });
        let ctx = sensor.sense().expect("sensor must return Some");
        // Without pool info, max_concurrency defaults to 1.0 → global_load = min(0.5, 1.0) = 0.5
        assert!(
            (ctx.global_load - 0.5).abs() < 1e-6,
            "global_load should be 0.5 (loadavg/1 concurrency)"
        );
    }

    #[test]
    fn loadavg_normalized_by_pool_concurrency() {
        let pool = PoolStats {
            pool_type: "threads".to_string(),
            max_concurrency: 4,
            pool_size: 4,
            available: 2,
            processes: alloc::vec![],
        };
        let sensor = CelersLoadSensor::new(move || {
            make_stats(
                100,
                2,
                98,
                0,
                Some([2.0, 1.5, 1.0]), // 1-min load = 2.0 with 4 threads → 0.5
                Some(pool.clone()),
                None,
            )
        });
        let ctx = sensor.sense().expect("sensor must return Some");
        assert!(
            (ctx.global_load - 0.5).abs() < 1e-6,
            "global_load should be 2/4 = 0.5"
        );
    }

    #[test]
    fn pool_available_maps_to_queue_depth() {
        let pool = PoolStats {
            pool_type: "prefork".to_string(),
            max_concurrency: 8,
            pool_size: 8,
            available: 3,
            processes: alloc::vec![1000, 1001, 1002],
        };
        let sensor =
            CelersLoadSensor::new(move || make_stats(50, 5, 45, 0, None, Some(pool.clone()), None));
        let ctx = sensor.sense().expect("sensor must return Some");
        // busy = pool_size - available = 8 - 3 = 5
        let queue_depth = ctx
            .queue_depth
            .get("__pool__")
            .copied()
            .expect("__pool__ key should exist");
        assert_eq!(queue_depth, 5, "pool queue_depth should be busy count (5)");
    }

    #[test]
    fn broker_connected_sets_circuit_closed() {
        use oxirouter::CircuitState;
        let broker = BrokerStats {
            url: "amqp://localhost:5672".to_string(),
            connected: true,
            heartbeat: Some(60),
            transport: "amqp".to_string(),
        };
        let sensor = CelersLoadSensor::new(move || {
            make_stats(20, 1, 19, 0, None, None, Some(broker.clone()))
        });
        let ctx = sensor.sense().expect("sensor must return Some");
        let state = ctx
            .circuit_breakers
            .get("__broker__")
            .copied()
            .expect("__broker__ circuit state should be set");
        assert_eq!(state, CircuitState::Closed);
        let health = ctx
            .target_health
            .get("__broker__")
            .copied()
            .expect("__broker__ health should be set");
        assert!((health - 1.0).abs() < 1e-6);
    }

    #[test]
    fn broker_disconnected_sets_circuit_open() {
        use oxirouter::CircuitState;
        let broker = BrokerStats {
            url: "amqp://remote:5672".to_string(),
            connected: false,
            heartbeat: None,
            transport: "amqp".to_string(),
        };
        let sensor =
            CelersLoadSensor::new(move || make_stats(0, 0, 0, 0, None, None, Some(broker.clone())));
        let ctx = sensor.sense().expect("sensor must return Some");
        let state = ctx
            .circuit_breakers
            .get("__broker__")
            .copied()
            .expect("__broker__ circuit state should be set");
        assert_eq!(state, CircuitState::Open);
    }

    #[test]
    fn error_rate_computed_from_failed_over_total() {
        // 10 failed out of 100 total → 0.1
        let sensor = CelersLoadSensor::new(|| make_stats(100, 0, 90, 10, None, None, None));
        let ctx = sensor.sense().expect("sensor must return Some");
        let error_rate = ctx
            .error_rates
            .get("__global__")
            .copied()
            .expect("__global__ error rate should be set");
        assert!((error_rate - 0.1).abs() < 1e-6, "error_rate should be 0.1");
    }

    #[test]
    fn zero_total_tasks_no_error_rate() {
        let sensor = CelersLoadSensor::new(|| make_stats(0, 0, 0, 0, None, None, None));
        let ctx = sensor.sense().expect("sensor must return Some");
        // When total_tasks == 0, we should NOT insert a __global__ error rate.
        assert!(
            ctx.error_rates.get("__global__").is_none(),
            "should not insert error rate when total_tasks == 0"
        );
    }

    #[test]
    fn global_load_clamps_to_one() {
        // loadavg = 5.0, concurrency = 1 → would be 5.0, clamp to 1.0
        let sensor =
            CelersLoadSensor::new(|| make_stats(10, 10, 0, 0, Some([5.0, 4.0, 3.0]), None, None));
        let ctx = sensor.sense().expect("sensor must return Some");
        assert!(
            (ctx.global_load - 1.0).abs() < 1e-6,
            "global_load must be clamped to 1.0"
        );
    }
}

extern crate alloc;
