//! Geographic context from oxigdal (Physical brain)

#[cfg(feature = "alloc")]
use alloc::string::String;

use serde::{Deserialize, Serialize};

/// Geographic context for location-aware routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeoContext {
    /// Current position as (longitude, latitude)
    pub position: Option<(f64, f64)>,

    /// ISO 3166-1 alpha-2 country code
    pub country_code: Option<String>,

    /// Region/state/province
    pub region: Option<String>,

    /// City name
    pub city: Option<String>,

    /// H3 or custom grid cell ID for spatial indexing
    pub grid_id: Option<u64>,

    /// Bounding box for area of interest [min_x, min_y, max_x, max_y]
    pub bbox: Option<[f64; 4]>,

    /// EPSG code for coordinate reference system
    pub epsg_code: u32,

    /// Accuracy of position in meters
    pub accuracy_meters: Option<f32>,

    /// Altitude in meters (if available)
    pub altitude_meters: Option<f32>,
}

impl GeoContext {
    /// Create a new geo context with WGS84 CRS
    #[must_use]
    pub const fn new() -> Self {
        Self {
            position: None,
            country_code: None,
            region: None,
            city: None,
            grid_id: None,
            bbox: None,
            epsg_code: 4326, // WGS84
            accuracy_meters: None,
            altitude_meters: None,
        }
    }

    /// Create geo context from coordinates
    #[must_use]
    pub const fn from_coords(lon: f64, lat: f64) -> Self {
        Self {
            position: Some((lon, lat)),
            country_code: None,
            region: None,
            city: None,
            grid_id: None,
            bbox: None,
            epsg_code: 4326,
            accuracy_meters: None,
            altitude_meters: None,
        }
    }

    /// Set the country code
    #[must_use]
    pub fn with_country(mut self, code: impl Into<String>) -> Self {
        self.country_code = Some(code.into());
        self
    }

    /// Set the region
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the bounding box
    #[must_use]
    pub const fn with_bbox(mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        self.bbox = Some([min_x, min_y, max_x, max_y]);
        self
    }

    /// Check if a point is within the bounding box
    #[must_use]
    pub fn contains_point(&self, lon: f64, lat: f64) -> bool {
        if let Some([min_x, min_y, max_x, max_y]) = self.bbox {
            lon >= min_x && lon <= max_x && lat >= min_y && lat <= max_y
        } else {
            false
        }
    }

    /// Calculate distance to another point in kilometers (Haversine formula)
    #[must_use]
    pub fn distance_to(&self, lon: f64, lat: f64) -> Option<f64> {
        let (self_lon, self_lat) = self.position?;

        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1 = self_lat.to_radians();
        let lat2 = lat.to_radians();
        let delta_lat = (lat - self_lat).to_radians();
        let delta_lon = (lon - self_lon).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);

        let c = 2.0 * a.sqrt().asin();

        Some(EARTH_RADIUS_KM * c)
    }

    /// Check if position is in EU/EEA (for GDPR)
    #[must_use]
    pub fn is_eu_region(&self) -> bool {
        const EU_COUNTRIES: &[&str] = &[
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
            // EEA
            "IS", "LI", "NO",
        ];

        if let Some(ref code) = self.country_code {
            EU_COUNTRIES.contains(&code.as_str())
        } else {
            false
        }
    }

    /// Get region tier for data residency (EU, US, APAC, other)
    #[must_use]
    pub fn data_residency_tier(&self) -> DataResidencyTier {
        if let Some(ref code) = self.country_code {
            match code.as_str() {
                "US" | "CA" => DataResidencyTier::Americas,
                "JP" | "CN" | "KR" | "SG" | "AU" | "NZ" | "IN" => DataResidencyTier::AsiaPacific,
                _ if self.is_eu_region() => DataResidencyTier::Europe,
                _ => DataResidencyTier::Other,
            }
        } else {
            DataResidencyTier::Unknown
        }
    }
}

/// Data residency tier for compliance routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataResidencyTier {
    /// European Union / EEA
    Europe,
    /// Americas (US, Canada)
    Americas,
    /// Asia-Pacific
    AsiaPacific,
    /// Other regions
    Other,
    /// Unknown/not specified
    Unknown,
}

/// Integration with oxigeo-core types
#[cfg(feature = "geo")]
impl GeoContext {
    /// Create from oxigdal BoundingBox
    #[must_use]
    pub fn from_oxigdal_bbox(bbox: &oxigeo_core::BoundingBox) -> Self {
        Self {
            position: Some((
                f64::midpoint(bbox.min_x, bbox.max_x),
                f64::midpoint(bbox.min_y, bbox.max_y),
            )),
            country_code: None,
            region: None,
            city: None,
            grid_id: None,
            bbox: Some([bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y]),
            epsg_code: 4326,
            accuracy_meters: None,
            altitude_meters: None,
        }
    }

    /// Convert to oxigdal BoundingBox
    #[must_use]
    pub fn to_oxigdal_bbox(&self) -> Option<oxigeo_core::BoundingBox> {
        if let Some([min_x, min_y, max_x, max_y]) = self.bbox {
            oxigeo_core::BoundingBox::new(min_x, min_y, max_x, max_y).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_context_creation() {
        let ctx = GeoContext::from_coords(-122.4194, 37.7749)
            .with_country("US")
            .with_region("California");

        assert_eq!(ctx.country_code, Some("US".to_string()));
        assert!(!ctx.is_eu_region());
    }

    #[test]
    fn test_distance_calculation() {
        let ctx = GeoContext::from_coords(0.0, 51.5); // London
        let distance = ctx.distance_to(-0.1276, 51.5074).unwrap();
        assert!(distance < 20.0); // Within 20km
    }

    #[test]
    fn test_eu_detection() {
        let ctx = GeoContext::new().with_country("DE");
        assert!(ctx.is_eu_region());

        let ctx = GeoContext::new().with_country("US");
        assert!(!ctx.is_eu_region());
    }

    #[test]
    fn test_data_residency() {
        assert_eq!(
            GeoContext::new().with_country("DE").data_residency_tier(),
            DataResidencyTier::Europe
        );
        assert_eq!(
            GeoContext::new().with_country("US").data_residency_tier(),
            DataResidencyTier::Americas
        );
        assert_eq!(
            GeoContext::new().with_country("JP").data_residency_tier(),
            DataResidencyTier::AsiaPacific
        );
    }
}
