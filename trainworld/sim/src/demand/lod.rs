//! T-1222: a zone's level of detail (notes/T-1206.md). One model at two sample densities: a
//! coarse zone draws fewer everyday destination samples and deals its small commute pairs in
//! bigger chunks; every budget stays exact and the rest of the model is the same.

use super::kernel::COMMUTE_CHUNK;

/// The levels a zone opens at (`DemandApi::at_level`): 0 fine, 1 coarse.
pub const FINE: u32 = 0;
pub const COARSE: u32 = 1;

/// How densely a cell's everyday destinations are sampled: one per `per` trips a day, rounded
/// up, from `min` to `max` (`max_few` for airports, medical and attractions, T-142).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub per: f64,
    pub min: usize,
    pub max: usize,
    pub max_few: usize,
    /// Mixed into every sample's seed: 0 in the game, other draws for the bias test.
    pub salt: u32,
}

impl Sampling {
    /// T-397's rule.
    pub const FINE: Sampling = Sampling { per: 8.0, min: 2, max: 16, max_few: 2, salt: 0 };
    /// T-1222's: a dense cell keeps up to 8, since 4 leaned a metro's rail up by about 0.5%
    /// (fewer samples to a subzone pair lean its rail up, free flow included); notes/T-1206.md.
    pub const COARSE: Sampling = Sampling { per: 64.0, min: 1, max: 8, max_few: 1, salt: 0 };
    pub fn count(&self, few: bool, trips: f32) -> usize {
        let most = if few { self.max_few } else { self.max };
        ((trips as f64 / self.per).ceil() as usize).clamp(self.min.min(most), most)
    }
}

/// What a level sets: the everyday sampling and `Params::commute_chunk` (T-874) while chunks
/// are on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lod {
    pub level: u32,
    pub sampling: Sampling,
    pub chunk: f32,
}

impl Lod {
    pub fn of(level: u32) -> Result<Lod, String> {
        match level {
            FINE => Ok(Lod { level, sampling: Sampling::FINE, chunk: COMMUTE_CHUNK }),
            COARSE => Ok(Lod { level, sampling: Sampling::COARSE, chunk: COARSE_CHUNK }),
            _ => Err(format!("no level {level}")),
        }
    }
}

/// A coarse zone's commute chunk (trips a period).
pub const COARSE_CHUNK: f32 = 1.0;
