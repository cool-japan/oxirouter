//! OxiGDAL-backed geographic sensor.
//!
//! Provides [`StaticOxigdalGeoSensor`] for fixed-position use cases and
//! [`DynamicOxigdalGeoSensor`] for runtime-updating position sources such
//! as GPS feeds on edge/drone deployments.

#[cfg(feature = "geo")]
use super::{GeoContext, sensor::GeoSensor};

/// A static geographic sensor backed by a pre-built `oxigdal_core::BoundingBox`.
///
/// Suitable for deployment-time configuration where the region of interest
/// is known at startup (data-center regional routing, fixed sensor nodes).
#[cfg(feature = "geo")]
pub struct StaticOxigdalGeoSensor {
    /// The bounding box representing the region of interest.
    pub bbox: oxigdal_core::BoundingBox,
    /// Optional ISO 3166-1 alpha-2 country code (e.g., `"DE"`, `"JP"`).
    pub country_code: Option<String>,
}

#[cfg(feature = "geo")]
impl GeoSensor for StaticOxigdalGeoSensor {
    fn sense(&self) -> Option<GeoContext> {
        let mut ctx = GeoContext::from_oxigdal_bbox(&self.bbox);
        if let Some(ref cc) = self.country_code {
            ctx = ctx.with_country(cc.clone());
        }
        Some(ctx)
    }
}

/// A dynamic geographic sensor that calls a closure to produce the bounding box.
///
/// Use this when position updates at runtime (e.g., drone GPS feed, vehicle
/// telemetry). The closure is called on every `sense()` invocation; heavy
/// I/O should be cached inside the closure itself.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "geo")]
/// # {
/// use oxirouter::context::DynamicOxigdalGeoSensor;
///
/// let sensor = DynamicOxigdalGeoSensor::new(|| {
///     // Replace with real GPS call
///     oxigdal_core::BoundingBox::new(13.0, 52.0, 14.0, 53.0).ok()
/// }).with_country_code("DE".to_string());
/// # }
/// ```
#[cfg(feature = "geo")]
pub struct DynamicOxigdalGeoSensor<F>
where
    F: Fn() -> Option<oxigdal_core::BoundingBox> + Send + Sync,
{
    /// Closure that yields the current bounding box, or `None` if unavailable.
    bbox_fn: F,
    /// Static country code hint applied alongside the dynamic bbox.
    country_code: Option<String>,
}

#[cfg(feature = "geo")]
impl<F> DynamicOxigdalGeoSensor<F>
where
    F: Fn() -> Option<oxigdal_core::BoundingBox> + Send + Sync,
{
    /// Create a new dynamic sensor with the given closure.
    #[must_use]
    pub fn new(bbox_fn: F) -> Self {
        Self {
            bbox_fn,
            country_code: None,
        }
    }

    /// Set a static country code hint (applied alongside the dynamic bbox).
    #[must_use]
    pub fn with_country_code(mut self, cc: String) -> Self {
        self.country_code = Some(cc);
        self
    }
}

#[cfg(feature = "geo")]
impl<F> GeoSensor for DynamicOxigdalGeoSensor<F>
where
    F: Fn() -> Option<oxigdal_core::BoundingBox> + Send + Sync,
{
    fn sense(&self) -> Option<GeoContext> {
        let bbox = (self.bbox_fn)()?;
        let mut ctx = GeoContext::from_oxigdal_bbox(&bbox);
        if let Some(ref cc) = self.country_code {
            ctx = ctx.with_country(cc.clone());
        }
        Some(ctx)
    }
}
