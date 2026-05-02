//! Device context from mielin (Body brain)

use serde::{Deserialize, Serialize};

/// Device/hardware context for resource-aware routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceContext {
    /// Battery percentage (0-100, None if plugged in or desktop)
    pub battery_pct: Option<u8>,

    /// Whether device is charging
    pub is_charging: bool,

    /// Network type (wifi, cellular, ethernet, etc.)
    pub network_type: NetworkType,

    /// Estimated bandwidth in kbps
    pub bandwidth_kbps: Option<u32>,

    /// Round-trip time in milliseconds
    pub rtt_ms: Option<u32>,

    /// CPU load percentage (0-100)
    pub cpu_load: Option<u8>,

    /// Available memory in MB
    pub available_memory_mb: Option<u32>,

    /// Device type
    pub device_type: DeviceType,

    /// Whether the device is in power saving mode
    pub power_saving: bool,

    /// Whether the device has metered connection
    pub metered_connection: bool,
}

impl DeviceContext {
    /// Create a new device context
    #[must_use]
    pub const fn new() -> Self {
        Self {
            battery_pct: None,
            is_charging: false,
            network_type: NetworkType::Unknown,
            bandwidth_kbps: None,
            rtt_ms: None,
            cpu_load: None,
            available_memory_mb: None,
            device_type: DeviceType::Unknown,
            power_saving: false,
            metered_connection: false,
        }
    }

    /// Set battery level
    #[must_use]
    pub fn with_battery(mut self, pct: u8) -> Self {
        self.battery_pct = Some(if pct > 100 { 100 } else { pct });
        self
    }

    /// Set network info
    #[must_use]
    pub const fn with_network(mut self, net_type: NetworkType, bandwidth_kbps: u32) -> Self {
        self.network_type = net_type;
        self.bandwidth_kbps = Some(bandwidth_kbps);
        self
    }

    /// Set RTT
    #[must_use]
    pub const fn with_rtt(mut self, rtt_ms: u32) -> Self {
        self.rtt_ms = Some(rtt_ms);
        self
    }

    /// Check if device is constrained (low battery, limited network, high CPU)
    #[must_use]
    pub fn is_constrained(&self) -> bool {
        // Low battery (under 20% and not charging)
        if let Some(battery) = self.battery_pct {
            if battery < 20 && !self.is_charging {
                return true;
            }
        }

        // Poor network
        if self.network_type == NetworkType::Cellular2G || self.network_type == NetworkType::Offline
        {
            return true;
        }

        // High CPU load
        if let Some(cpu) = self.cpu_load {
            if cpu > 80 {
                return true;
            }
        }

        // Power saving mode
        if self.power_saving {
            return true;
        }

        false
    }

    /// Get network quality score (0.0 - 1.0)
    #[must_use]
    pub fn network_quality(&self) -> f32 {
        let type_score = match self.network_type {
            NetworkType::Ethernet => 1.0,
            NetworkType::Wifi => 0.9,
            NetworkType::Cellular5G => 0.85,
            NetworkType::Cellular4G => 0.7,
            NetworkType::Cellular3G => 0.4,
            NetworkType::Cellular2G => 0.2,
            NetworkType::Satellite => 0.3,
            NetworkType::Offline => 0.0,
            NetworkType::Unknown => 0.5,
        };

        let bandwidth_score = if let Some(bw) = self.bandwidth_kbps {
            (bw as f32 / 100_000.0).min(1.0) // Normalize to 100 Mbps
        } else {
            0.5
        };

        let rtt_score = if let Some(rtt) = self.rtt_ms {
            (1.0 - (rtt as f32 / 1000.0)).max(0.0) // Lower RTT is better
        } else {
            0.5
        };

        // Weighted average
        type_score * 0.4 + bandwidth_score * 0.3 + rtt_score * 0.3
    }

    /// Get resource availability score (0.0 - 1.0)
    #[must_use]
    pub fn resource_availability(&self) -> f32 {
        let mut score = 1.0;

        // Battery impact
        if let Some(battery) = self.battery_pct {
            if !self.is_charging {
                score *= battery as f32 / 100.0;
            }
        }

        // CPU impact
        if let Some(cpu) = self.cpu_load {
            score *= (100 - cpu) as f32 / 100.0;
        }

        // Memory impact
        if let Some(mem) = self.available_memory_mb {
            // Assume 100MB is minimum needed
            score *= (mem as f32 / 1000.0).min(1.0);
        }

        // Power saving mode halves available resources
        if self.power_saving {
            score *= 0.5;
        }

        score
    }

    /// Returns `true` when the device has no usable network connection.
    ///
    /// A device is considered offline when its `network_type` is
    /// [`NetworkType::Offline`] **or** when `bandwidth_kbps` is explicitly
    /// set to `0`.
    #[must_use]
    pub fn is_offline(&self) -> bool {
        self.network_type == NetworkType::Offline || self.bandwidth_kbps == Some(0)
    }

    /// Recommended maximum result size based on constraints
    #[must_use]
    pub fn recommended_max_results(&self) -> u32 {
        if self.is_constrained() {
            100
        } else if self.metered_connection {
            500
        } else {
            match self.device_type {
                DeviceType::Server => 100_000,
                DeviceType::Desktop => 10_000,
                DeviceType::Laptop => 5_000,
                DeviceType::Tablet => 1_000,
                DeviceType::Phone => 500,
                DeviceType::Embedded => 100,
                DeviceType::Unknown => 1_000,
            }
        }
    }
}

/// Type of network connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NetworkType {
    /// Wired ethernet
    Ethernet,
    /// WiFi connection
    Wifi,
    /// 5G cellular
    Cellular5G,
    /// 4G/LTE cellular
    Cellular4G,
    /// 3G cellular
    Cellular3G,
    /// 2G/EDGE cellular
    Cellular2G,
    /// Satellite connection
    Satellite,
    /// No network connection
    Offline,
    /// Unknown network type
    #[default]
    Unknown,
}

/// Type of device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DeviceType {
    /// Server or datacenter
    Server,
    /// Desktop computer
    Desktop,
    /// Laptop computer
    Laptop,
    /// Tablet device
    Tablet,
    /// Smartphone
    Phone,
    /// Embedded/IoT device
    Embedded,
    /// Unknown device type
    #[default]
    Unknown,
}

/// Integration with mielin-hal
#[cfg(feature = "device")]
impl DeviceContext {
    /// Create from mielin HAL hardware detection.
    ///
    /// Populates CPU core count (as `cpu_load` proxy), available memory, and
    /// device type from platform detection. Battery and network fields remain
    /// `None`/`Unknown` because `mielin_hal` does not expose those interfaces —
    /// they require OS or browser-specific APIs layered above the HAL.
    pub fn from_mielin_hal() -> Self {
        use mielin_hal::{
            capabilities::HardwareProfile, platform::detect_platform, system::MemoryInfo,
        };

        let profile = HardwareProfile::detect();
        let mem = MemoryInfo::detect();
        let platform = detect_platform();

        // Map core count to a rough cpu_load proxy:
        // We can't measure actual CPU utilisation from the HAL, so we leave
        // cpu_load as None and only populate structural device information.
        // Callers that need live CPU load should layer a sysinfo sensor.

        let available_memory_mb = if mem.available > 0 {
            // Convert bytes -> MB, clamp to u32::MAX
            let mb = mem.available / (1024 * 1024);
            Some(mb.min(u32::MAX as usize) as u32)
        } else if profile.memory_size > 0 {
            // Fall back to total memory if available is not detectable
            let mb = profile.memory_size / (1024 * 1024);
            Some(mb.min(u32::MAX as usize) as u32)
        } else {
            None
        };

        let device_type = Self::device_type_from_platform(&platform);

        Self {
            battery_pct: None,                  // mielin_hal does not expose battery status
            is_charging: false,                 // unknown; default to false (conservative)
            network_type: NetworkType::Unknown, // mielin_hal has no network APIs
            bandwidth_kbps: None,
            rtt_ms: None,
            cpu_load: None, // live CPU load not available from HAL
            available_memory_mb,
            device_type,
            power_saving: false,
            metered_connection: false,
        }
    }

    /// Map `mielin_hal::Platform` to `DeviceType`.
    fn device_type_from_platform(platform: &mielin_hal::platform::Platform) -> DeviceType {
        use mielin_hal::platform::Platform;
        match platform {
            Platform::RaspberryPi(_) | Platform::BeagleBone(_) => DeviceType::Embedded,
            Platform::Stm32(_) | Platform::Esp32(_) => DeviceType::Embedded,
            Platform::Jetson(_) => DeviceType::Desktop,
            Platform::GenericX86 => DeviceType::Desktop,
            Platform::GenericArm => DeviceType::Server,
            Platform::GenericRiscV => DeviceType::Server,
            Platform::Unknown => DeviceType::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_context() {
        let ctx = DeviceContext::new()
            .with_battery(50)
            .with_network(NetworkType::Wifi, 50_000)
            .with_rtt(50);

        assert!(!ctx.is_constrained());
        assert!(ctx.network_quality() > 0.5);
    }

    #[test]
    fn test_constrained_detection() {
        let low_battery = DeviceContext::new().with_battery(10);
        assert!(low_battery.is_constrained());

        let charging = DeviceContext {
            battery_pct: Some(10),
            is_charging: true,
            ..Default::default()
        };
        assert!(!charging.is_constrained());
    }

    #[test]
    fn test_network_quality() {
        let ethernet = DeviceContext {
            network_type: NetworkType::Ethernet,
            bandwidth_kbps: Some(1_000_000),
            rtt_ms: Some(5),
            ..Default::default()
        };
        assert!(ethernet.network_quality() > 0.9);

        let poor_2g = DeviceContext {
            network_type: NetworkType::Cellular2G,
            bandwidth_kbps: Some(100),
            rtt_ms: Some(500),
            ..Default::default()
        };
        assert!(poor_2g.network_quality() < 0.3);
    }

    #[test]
    fn test_recommended_results() {
        let server = DeviceContext {
            device_type: DeviceType::Server,
            ..Default::default()
        };
        assert!(server.recommended_max_results() >= 10_000);

        let constrained_phone = DeviceContext {
            device_type: DeviceType::Phone,
            battery_pct: Some(5),
            is_charging: false,
            ..Default::default()
        };
        assert!(constrained_phone.recommended_max_results() <= 500);
    }
}
