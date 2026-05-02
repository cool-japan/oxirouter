//! Tests for OxiGDAL-backed geographic sensor.

#[cfg(feature = "geo")]
mod geo_sensor_tests {
    use oxirouter::{DynamicOxigdalGeoSensor, StaticOxigdalGeoSensor, context::sensor::GeoSensor};

    fn bbox(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> oxigdal_core::BoundingBox {
        oxigdal_core::BoundingBox::new(min_x, min_y, max_x, max_y)
            .expect("valid bounding box coordinates")
    }

    #[test]
    fn static_sensor_round_trips_bbox() {
        let b = bbox(8.0, 47.0, 15.0, 55.0); // rough Germany
        let sensor = StaticOxigdalGeoSensor {
            bbox: b,
            country_code: Some("DE".to_string()),
        };

        let ctx = sensor.sense().expect("static sensor must return Some");

        // Center should be midpoint of bbox
        let expected_lon = f64::midpoint(8.0, 15.0);
        let expected_lat = f64::midpoint(47.0, 55.0);
        let (lon, lat) = ctx.position.expect("position should be Some");

        assert!(
            (lon - expected_lon).abs() < 1e-10,
            "longitude center mismatch: {} vs {}",
            lon,
            expected_lon
        );
        assert!(
            (lat - expected_lat).abs() < 1e-10,
            "latitude center mismatch: {} vs {}",
            lat,
            expected_lat
        );
        assert_eq!(ctx.country_code.as_deref(), Some("DE"));
    }

    #[test]
    fn static_sensor_bbox_preserved() {
        let b = bbox(-180.0, -90.0, 180.0, 90.0); // whole world
        let sensor = StaticOxigdalGeoSensor {
            bbox: b,
            country_code: None,
        };

        let ctx = sensor.sense().expect("static sensor must return Some");
        let stored = ctx.bbox.expect("bbox should be Some");

        assert_eq!(stored, [-180.0, -90.0, 180.0, 90.0]);
        assert!(ctx.country_code.is_none());
    }

    #[test]
    fn static_sensor_no_country_code() {
        let b = bbox(0.0, 0.0, 1.0, 1.0);
        let sensor = StaticOxigdalGeoSensor {
            bbox: b,
            country_code: None,
        };

        let ctx = sensor.sense().expect("must return Some");
        assert!(ctx.country_code.is_none());
    }

    #[test]
    fn dynamic_sensor_returns_none_when_closure_returns_none() {
        let sensor = DynamicOxigdalGeoSensor::new(|| None);
        assert!(
            sensor.sense().is_none(),
            "sense() should be None when closure returns None"
        );
    }

    #[test]
    fn dynamic_sensor_round_trips_bbox() {
        let sensor = DynamicOxigdalGeoSensor::new(|| {
            oxigdal_core::BoundingBox::new(139.0, 35.0, 140.0, 36.0).ok()
        })
        .with_country_code("JP".to_string());

        let ctx = sensor.sense().expect("dynamic sensor must return Some");
        let expected_lon = f64::midpoint(139.0, 140.0);
        let expected_lat = f64::midpoint(35.0, 36.0);
        let (lon, lat) = ctx.position.expect("position should be Some");

        assert!((lon - expected_lon).abs() < 1e-10);
        assert!((lat - expected_lat).abs() < 1e-10);
        assert_eq!(ctx.country_code.as_deref(), Some("JP"));
    }

    #[test]
    fn dynamic_sensor_without_country_code() {
        let sensor = DynamicOxigdalGeoSensor::new(|| {
            oxigdal_core::BoundingBox::new(-74.0, 40.0, -73.0, 41.0).ok()
        });

        let ctx = sensor.sense().expect("dynamic sensor must return Some");
        assert!(ctx.country_code.is_none());
        let (lon, lat) = ctx.position.expect("position should be Some");
        assert!(lon < -73.0 && lon > -74.0);
        assert!(lat < 41.0 && lat > 40.0);
    }

    #[test]
    fn roundtrip_to_oxigdal_bbox() {
        let original = bbox(10.0, 50.0, 15.0, 55.0);
        let sensor = StaticOxigdalGeoSensor {
            bbox: original,
            country_code: None,
        };

        let ctx = sensor.sense().expect("must return Some");
        let roundtrip = ctx
            .to_oxigdal_bbox()
            .expect("to_oxigdal_bbox must return Some");

        // The roundtrip bbox should match the original coordinates.
        let re_ctx = oxirouter::GeoContext::from_oxigdal_bbox(&roundtrip);
        let stored = re_ctx.bbox.expect("bbox should survive roundtrip");
        assert_eq!(stored, [10.0, 50.0, 15.0, 55.0]);
    }
}
