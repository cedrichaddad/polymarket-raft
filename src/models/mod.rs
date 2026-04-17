//! Model layer (§11).
//!
//!  * `fair_value` — parametric p_0(t) = Phi((log(S_t/K) - b) / sigma)
//!  * `calibrator` — monotone (isotonic) calibration p_0 -> p*
//!  * `retrieval` — nearest-neighbor retrieval trait used by the Vibrato
//!                   integration (§17)

pub mod fair_value;
pub mod calibrator;
pub mod retrieval;

pub use fair_value::{fair_prob_backbone, FairValueParams};
pub use calibrator::{Calibrator, IsotonicCalibrator};
pub use retrieval::{Retrieval, RetrievalHit, RetrievalQuery};
