use chrono::{DateTime, Utc};
use gea_inference::InferenceResult;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditPhase {
    Ingest,
    FeatureExtract,
    Infer,
    Validate,
    Act,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub phase: AuditPhase,
    pub agent: String,
    pub action: String,
    pub inference_id: Option<Uuid>,
    pub details: serde_json::Value,
    pub content_hash: String,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceAuditEvent {
    pub trace: TraceEvent,
    pub inference: InferenceResult,
    pub policy_passed: bool,
}

#[derive(Debug, Clone)]
pub struct GovernanceConfig {
    pub signing_key: Option<String>,
    pub ledger_path: PathBuf,
    pub confidence_floor: f32,
}

#[derive(Debug, Error)]
pub enum AuditError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct AuditLedger {
    config: GovernanceConfig,
}

impl AuditLedger {
    pub fn new(config: GovernanceConfig) -> Self {
        Self { config }
    }

    pub fn record_inference(
        &self,
        inference: &InferenceResult,
        policy_passed: bool,
    ) -> Result<InferenceAuditEvent, AuditError> {
        let details = serde_json::json!({
            "rack_id": inference.rack_id,
            "module_index": inference.module_index,
            "fault_class": inference.fault_class,
            "confidence": inference.confidence,
            "severity": inference.severity,
            "recommended_action": inference.recommended_action,
            "latency_us": inference.latency_us,
            "precision": inference.precision,
            "policy_passed": policy_passed,
        });

        let trace = self.record(
            AuditPhase::Infer,
            "gea-edge-agent",
            "classify_fault",
            Some(inference.inference_id),
            details,
        )?;

        let event = InferenceAuditEvent {
            trace,
            inference: inference.clone(),
            policy_passed,
        };
        Ok(event)
    }

    pub fn record(
        &self,
        phase: AuditPhase,
        agent: &str,
        action: &str,
        inference_id: Option<Uuid>,
        details: serde_json::Value,
    ) -> Result<TraceEvent, AuditError> {
        let timestamp = Utc::now();
        let canonical = serde_json::json!({
            "phase": phase,
            "agent": agent,
            "action": action,
            "inference_id": inference_id,
            "details": details,
            "timestamp": timestamp.to_rfc3339(),
        });
        let content_hash = hash_payload(&canonical);
        let signature = self
            .config
            .signing_key
            .as_ref()
            .map(|key| sign_payload(key, &content_hash));

        let event = TraceEvent {
            id: Uuid::new_v4(),
            timestamp,
            phase,
            agent: agent.to_string(),
            action: action.to_string(),
            inference_id,
            details,
            content_hash,
            signature,
        };
        self.append(&event)?;
        Ok(event)
    }

    pub fn read_all(&self) -> Result<Vec<TraceEvent>, AuditError> {
        if !self.config.ledger_path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.config.ledger_path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            events.push(serde_json::from_str(&line)?);
        }
        Ok(events)
    }

    pub fn passes_policy(&self, inference: &InferenceResult) -> bool {
        if inference.fault_class.is_fault() && inference.confidence < self.config.confidence_floor {
            return false;
        }
        true
    }

    fn append(&self, event: &TraceEvent) -> Result<(), AuditError> {
        if let Some(parent) = self.config.ledger_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.ledger_path)?;
        writeln!(file, "{}", serde_json::to_string(event)?)?;
        Ok(())
    }
}

pub fn hash_payload(value: &serde_json::Value) -> String {
    hex::encode(Sha256::digest(
        serde_json::to_vec(value).unwrap_or_default(),
    ))
}

fn sign_payload(key: &str, content_hash: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(key.as_bytes()).expect("hmac");
    mac.update(content_hash.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

pub fn default_ledger_path(root: &Path) -> PathBuf {
    root.join(".gea").join("audit.jsonl")
}
