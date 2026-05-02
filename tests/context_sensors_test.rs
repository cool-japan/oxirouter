//! Tests for the sensor trait infrastructure and `EcosystemContextProvider`.
//!
//! Uses fake sensors to verify context population, caching, and cache invalidation.

#[cfg(feature = "ecosystem")]
mod ecosystem_tests {
    use oxirouter::{
        DeviceContext, GeoContext, LegalContext, LoadContext, NetworkType,
        context::{ContextProvider, EcosystemContextProvider},
        sensor::{DeviceSensor, GeoSensor, LoadSensor, PolicyEngine},
    };

    // ----- Fake geo sensor -----

    struct FixedGeoSensor {
        ctx: GeoContext,
    }

    impl GeoSensor for FixedGeoSensor {
        fn sense(&self) -> Option<GeoContext> {
            Some(self.ctx.clone())
        }
    }

    // ----- Fake device sensor -----

    struct FixedDeviceSensor {
        ctx: DeviceContext,
    }

    impl DeviceSensor for FixedDeviceSensor {
        fn sense(&self) -> Option<DeviceContext> {
            Some(self.ctx.clone())
        }
    }

    // ----- Fake load sensor -----

    struct FixedLoadSensor {
        ctx: LoadContext,
    }

    impl LoadSensor for FixedLoadSensor {
        fn sense(&self) -> Option<LoadContext> {
            Some(self.ctx.clone())
        }
    }

    // ----- Fake policy engine -----

    struct FixedPolicyEngine {
        ctx: LegalContext,
    }

    impl PolicyEngine for FixedPolicyEngine {
        fn evaluate_for_jurisdiction(&self, _jurisdiction: &str) -> Option<LegalContext> {
            Some(self.ctx.clone())
        }
    }

    // ----- Returning-None sensors -----

    struct NeverGeoSensor;
    impl GeoSensor for NeverGeoSensor {
        fn sense(&self) -> Option<GeoContext> {
            None
        }
    }

    struct NeverDeviceSensor;
    impl DeviceSensor for NeverDeviceSensor {
        fn sense(&self) -> Option<DeviceContext> {
            None
        }
    }

    struct NeverLoadSensor;
    impl LoadSensor for NeverLoadSensor {
        fn sense(&self) -> Option<LoadContext> {
            None
        }
    }

    struct NeverPolicyEngine;
    impl PolicyEngine for NeverPolicyEngine {
        fn evaluate_for_jurisdiction(&self, _jurisdiction: &str) -> Option<LegalContext> {
            None
        }
    }

    // ----- Helper -----

    fn geo_sensor_with_coords(lon: f64, lat: f64, country: &str) -> GeoContext {
        GeoContext::from_coords(lon, lat).with_country(country.to_string())
    }

    // ----- Tests -----

    #[test]
    fn fake_geo_sensor_populates_context() {
        let geo = geo_sensor_with_coords(13.404_954, 52.520_008, "DE");
        let provider = EcosystemContextProvider::new()
            .with_geo_sensor(Box::new(FixedGeoSensor { ctx: geo.clone() }));

        let combined = provider.get_combined_context();
        let returned_geo = combined.geo.expect("geo context should be Some");
        assert_eq!(returned_geo.country_code.as_deref(), Some("DE"));
        assert!(returned_geo.position.is_some());
    }

    #[test]
    fn fake_device_sensor_populates_context() {
        let device = DeviceContext::new()
            .with_battery(75)
            .with_network(NetworkType::Wifi, 50_000)
            .with_rtt(20);

        let provider = EcosystemContextProvider::new()
            .with_device_sensor(Box::new(FixedDeviceSensor { ctx: device }));

        let combined = provider.get_combined_context();
        let returned_device = combined.device.expect("device context should be Some");
        assert_eq!(returned_device.battery_pct, Some(75));
    }

    #[test]
    fn fake_load_sensor_populates_context() {
        let mut load = LoadContext::new();
        load.global_load = 0.42;
        load.pending_tasks = 7;

        let provider = EcosystemContextProvider::new()
            .with_load_sensor(Box::new(FixedLoadSensor { ctx: load }));

        let combined = provider.get_combined_context();
        let returned_load = combined.load.expect("load context should be Some");
        assert!((returned_load.global_load - 0.42).abs() < 1e-6);
        assert_eq!(returned_load.pending_tasks, 7);
    }

    #[test]
    fn fake_policy_engine_populates_context() {
        let legal = LegalContext::gdpr();

        let provider = EcosystemContextProvider::new()
            .with_policy_engine(Box::new(FixedPolicyEngine { ctx: legal }));

        let combined = provider.get_combined_context();
        let returned_legal = combined.legal.expect("legal context should be Some");
        assert!(returned_legal.gdpr_region);
        assert!(returned_legal.audit_required);
    }

    #[test]
    fn cache_hit_on_second_call() {
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        struct CountingGeoSensor {
            counter: std::sync::Arc<std::sync::atomic::AtomicU32>,
        }

        impl GeoSensor for CountingGeoSensor {
            fn sense(&self) -> Option<GeoContext> {
                self.counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(GeoContext::from_coords(0.0, 0.0).with_country("JP".to_string()))
            }
        }

        let provider = EcosystemContextProvider::new()
            .with_ttl(3600) // very long TTL — cache should be hit on second call
            .with_geo_sensor(Box::new(CountingGeoSensor {
                counter: call_count_clone,
            }));

        let _first = provider.get_combined_context();
        let _second = provider.get_combined_context();

        // The sensor should have been called exactly once (second call hits cache).
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "sensor should only be called once when cache is warm"
        );
    }

    #[test]
    fn invalidate_cache_forces_recompute() {
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        struct CountingGeoSensor2 {
            counter: std::sync::Arc<std::sync::atomic::AtomicU32>,
        }

        impl GeoSensor for CountingGeoSensor2 {
            fn sense(&self) -> Option<GeoContext> {
                self.counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(GeoContext::from_coords(1.0, 1.0).with_country("FR".to_string()))
            }
        }

        let provider = EcosystemContextProvider::new()
            .with_ttl(3600)
            .with_geo_sensor(Box::new(CountingGeoSensor2 {
                counter: call_count_clone,
            }));

        let _first = provider.get_combined_context();
        provider.invalidate_cache();
        let _second = provider.get_combined_context();

        // After invalidation the sensor must be called again.
        assert_eq!(
            call_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "sensor should be called twice after cache invalidation"
        );
    }

    #[test]
    fn null_sensors_return_none_fields() {
        let provider = EcosystemContextProvider::new()
            .with_geo_sensor(Box::new(NeverGeoSensor))
            .with_device_sensor(Box::new(NeverDeviceSensor))
            .with_load_sensor(Box::new(NeverLoadSensor))
            .with_policy_engine(Box::new(NeverPolicyEngine));

        let combined = provider.get_combined_context();
        assert!(
            combined.geo.is_none(),
            "geo should be None for NeverGeoSensor"
        );
        assert!(
            combined.device.is_none(),
            "device should be None for NeverDeviceSensor"
        );
        assert!(
            combined.load.is_none(),
            "load should be None for NeverLoadSensor"
        );
        assert!(
            combined.legal.is_none(),
            "legal should be None for NeverPolicyEngine"
        );
    }

    #[test]
    fn no_sensors_returns_all_none() {
        let provider = EcosystemContextProvider::new();
        let combined = provider.get_combined_context();
        assert!(combined.geo.is_none());
        assert!(combined.device.is_none());
        assert!(combined.load.is_none());
        assert!(combined.legal.is_none());
    }
}
